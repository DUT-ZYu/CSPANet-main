### Art-fid
run:
```
cd evaluation;
python eval_artfid.py --sty ../results/Mystyle/sty --cnt ../results/Mystyle/con --tar ../results/Mystyle/img
                              #风格图像路径               内容图像路径       生成图像路径
```

### Histogram loss
run:
```
cd evaluation;
python eval_histogan.py --sty ../data/sty_eval --tar ../output
```
