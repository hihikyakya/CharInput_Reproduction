"""Optional fresh CE implementation. Run --help without installing torch/spaCy."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

from data import CONTEXT, EOS, PAD, UNK, VOCAB_SIZE, char_row, make_vocab, partition, passages, shifted


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wiki', type=Path, help='Local WikiText-103 train file (UTF-8)')
    p.add_argument('--lambada', type=Path, help='Optional local LAMBADA test .txt or .jsonl file')
    p.add_argument('--out', type=Path, default=Path('ce_run'))
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--smoke', action='store_true', help='Test both models without any external dataset')
    args = p.parse_args()
    if not args.smoke and not args.wiki:
        p.error('--wiki is required unless --smoke is used')
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('seeds must be distinct nonnegative integers')
    if args.epochs < 1 or args.batch < 1 or not 0 <= args.dropout < 1:
        p.error('epochs/batch must be positive and dropout in [0,1).')
    return args


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from model import CausalLM

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.smoke:
        from model import smoke
        smoke(device)
        return
    import spacy
    # Fail if a previous run uses this destination. No partial run is silently reused.
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = spacy.blank('en').tokenizer
    def tokenize(line):
        return [t.text for t in tokenizer(line) if not t.is_space]
    lines, tr, va = partition(args.wiki, args.split_seed)
    words = [tokenize(line) for line in lines]
    vocab = make_vocab(words[i] for i in tr)
    save_json(args.out / 'vocab.json', vocab)
    save_json(args.out / 'split.json', {'train': tr, 'validation': va})
    # One internal key per observed spelling serves only as a cache index. Both trained
    # lookup IDs and output classes are assigned exclusively by the training vocabulary.
    spellings = {PAD: 0}
    def surface_key(s):
        if s not in spellings:
            spellings[s] = len(spellings)
        return spellings[s]
    def encode(i):
        ctx, target = shifted(words[i]); return ([surface_key(s) for s in ctx], [vocab.get(s, vocab[UNK]) for s in target])
    train_data = [encode(i) for i in tr if words[i]]
    val_data = [encode(i) for i in va if words[i]]
    tests = []
    if args.lambada:
        for prefix, last in passages(args.lambada):
            ctx = tokenize(prefix)[-CONTEXT:]
            if not ctx:
                raise ValueError('LAMBADA passage has an empty tokenized context.')
            tests.append(([surface_key(s) for s in ctx], vocab.get(last, vocab[UNK]), last in vocab and last != UNK))
    if args.lambada and not tests:
        raise ValueError('The provided LAMBADA file contains no passages.')
    ids = torch.tensor([vocab.get(s, vocab[UNK]) for s in spellings], device=device)
    chars = torch.tensor([char_row(s) if s != PAD else [0]*128 for s in spellings], dtype=torch.uint8, device=device)
    def batch_context(rows):
        width = max(map(len, rows))
        values = [r + [0] * (width - len(r)) for r in rows]
        return torch.tensor(values, dtype=torch.long, device=device)
    def iterate(data, order):
        for offset in range(0, len(order), args.batch):
            rows = [data[i] for i in order[offset:offset + args.batch]]
            yield batch_context([r[0] for r in rows]), batch_context([r[1] for r in rows])
    def epoch(model, data, order, opt=None):
        model.train(opt is not None)
        total_loss, n = 0., 0
        with torch.set_grad_enabled(opt is not None):
            for x, y in iterate(data, order):
                valid = y != vocab[PAD]
                hidden = model(x, ids, chars)
                logits = model.out(hidden[valid]); truth = y[valid]
                loss = F.cross_entropy(logits, truth, reduction='sum')
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                    (loss / truth.numel()).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.detach().item(); n += truth.numel()
        return total_loss/n
    @torch.no_grad()
    def evaluate(model):
        model.eval()
        loss = [0., 0.]; count = [0, 0]; correct_iv = 0
        for start in range(0, len(tests), args.batch):
            block = tests[start:start+args.batch]; x = batch_context([r[0] for r in block]); h = model(x, ids, chars)
            positions = torch.tensor([len(r[0])-1 for r in block], device=device)
            logits = model.out(h[torch.arange(len(block), device=device), positions]); pred = logits.argmax(dim=1).tolist()
            truth = torch.tensor([r[1] for r in block], device=device)
            nll = F.cross_entropy(logits, truth, reduction='none').tolist()
            for (_, label, is_iv), xent, guess in zip(block, nll, pred):
                group = int(is_iv); count[group] += 1; loss[group] += xent
                correct_iv += int(is_iv and guess == label)
        n = sum(count)
        return {'passages': n, 'IV': count[1], 'OOV': count[0], 'mapped_CE_all': sum(loss)/n,
                'mapped_CE_OOV': loss[0]/count[0] if count[0] else None,
                'CE_IV': loss[1]/count[1] if count[1] else None,
                'literal_top1_IV': correct_iv/count[1] if count[1] else None, 'literal_top1_all': correct_iv/n}
    cfg = vars(args).copy()
    for field in ('out','wiki','lambada'):
        cfg[field] = str(cfg[field]) if cfg[field] is not None else None
    cfg.update(torch=str(torch.__version__), spacy=spacy.__version__, numpy=np.__version__, python=platform.python_version(),
               resolved_device=str(device), wiki_sha256=file_hash(args.wiki), lambada_sha256=file_hash(args.lambada) if args.lambada else None,
               effective_vocab=len(vocab), output_rows=VOCAB_SIZE)
    save_json(args.out / 'config.json', cfg)
    paired = []
    for seed in args.seeds:
        reports = {}
        for branch in ('lookup','character'):
            torch.manual_seed(seed)
            model = CausalLM(branch, args.dropout).to(device)
            params = sum(p.numel() for p in model.parameters())
            assert params == {'lookup':6_726_176,'character':4_230_145}[branch]
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
            run_dir = args.out/f'{seed}_{branch}'; run_dir.mkdir()
            learning_curve = []; best = math.inf; best_epoch = None
            for k in range(1, args.epochs+1):
                # Same order for both branches; independent model runs use identical initial shared layers.
                order = np.random.default_rng([seed, k]).permutation(len(train_data)).tolist()
                torch.manual_seed(seed + 100_003*k)
                train_ce = epoch(model, train_data, order, opt)
                val_ce = epoch(model, val_data, range(len(val_data)))
                if not math.isfinite(val_ce):
                    raise ArithmeticError('Nonfinite validation CE')
                learning_curve.append({'epoch':k, 'train_CE_online':train_ce,'validation_CE':val_ce})
                save_json(run_dir/'curve.json',learning_curve)
                print(f'{branch} seed={seed} epoch={k} CE(train/valid)={train_ce:.4f}/{val_ce:.4f}',flush=True)
                if val_ce < best:
                    best, best_epoch = val_ce, k
                    torch.save(model.state_dict(), run_dir/'best.pt')
            torch.save(model.state_dict(), run_dir/'last.pt')
            report = {'best_epoch':best_epoch,'best_val_CE':best,'last_val_CE':val_ce, 'params':params}
            if tests:
                report['LAMBADA_last'] = evaluate(model)
                model.load_state_dict(torch.load(run_dir/'best.pt',weights_only=True,map_location=device))
                report['LAMBADA_best'] = evaluate(model)
            save_json(run_dir/'report.json', report)
            reports[branch] = report
            del opt, model
        item = {'seed':seed}
        for metric in ('best_val_CE','last_val_CE'):
            item[metric] = reports['character'][metric] - reports['lookup'][metric]
        paired.append(item)
    aggregate = {}
    for metric in ('best_val_CE','last_val_CE'):
        values = [r[metric] for r in paired]
        aggregate[metric] = {'mean_delta': float(np.mean(values)),
                             'sample_sd_delta': float(np.std(values, ddof=1)) if len(values)>1 else None}
    save_json(args.out/'paired.json', {'sign':'character minus lookup; CE nats per valid token',
                                     'per_seed':paired, 'summary':aggregate})

if __name__ == '__main__':
    run(cli())
