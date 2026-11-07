"""Offline checks of split/token rules and expected parameter counts; no torch required."""
from pathlib import Path
from tempfile import TemporaryDirectory
from data import CHAR, SPECIAL, WIDTH, EOS, UNK, VOCAB_SIZE, char_row, make_vocab, partition, passages, shifted


def size():
    d = WIDTH; inner = 512
    # QKV + output attention matrices and biases, 2 FF projections, two affine LNs
    backbone = 8 * (4*d*d + 4*d + d*inner+inner + inner*d+d + 4*d)
    output = d*VOCAB_SIZE+VOCAB_SIZE
    enc = 70*d; old = d
    for c in (64,32,16):
        projection = old*c+c + 2*c
        residual = (3*c*c+c)+(c*c+c)+2*c
        enc += projection + 2*residual; old = c
    enc += 16 + 1
    assert (backbone, output, enc) == (1_586_176, 2_580_000, 63_969)
    total = (VOCAB_SIZE*d+output+backbone, enc+output+backbone)
    assert total == (6_726_176, 4_230_145)
    return total


def main():
    assert len(CHAR)==69 and sorted(CHAR.values())==list(range(1,70))
    assert char_row('CaT')==char_row('cat') and char_row('ü')[0]==0
    assert len(char_row('a'*200))==128 and char_row('A')[1]==0
    vocab=make_vocab([['Cat','cat','Cat']]); assert vocab['Cat']!=vocab['cat'] and vocab.get('new',vocab[UNK])==1
    assert len(SPECIAL)==6; assert shifted(['a','B'])==(['a','B'],['B',EOS])
    x,y=shifted([f'w{k}' for k in range(80)]); assert len(x)==len(y)==64 and y[-1]=='w64'
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'t.txt'; path.write_text('\n'+'\n'.join(f'topic {i}' for i in range(5007)),encoding='utf-8')
        lines,a,b=partition(path,2026); assert (len(lines),len(a),len(b))==(5000,3500,1500)
        assert not set(a)&set(b) and a==partition(path,2026)[1]
        path.write_text('My name is Kim\n',encoding='utf-8'); assert list(passages(path))==[['My name is','Kim']]
        qp=Path(tmp)/'q.jsonl'; qp.write_text('{"text":"a path ends"}\n',encoding='utf-8'); assert list(passages(qp))==[['a path','ends']]
    totals=size(); print(f'Protocol checks passed. Independent parameter totals: lookup={totals[0]:,}; character={totals[1]:,}.')

if __name__=='__main__':
    main()
