FLEXGEN_PATH=$PWD/../../flexgen
rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py


CMD="--model facebook/opt-13b --path /home/liushenyu/opt-13b-flexgen-np/ --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"
CMD=$CMD" --alpha 5 --partial-weight-ratio 0.2 --max-num-kv 409 --prefetch-threshold 20"
python -m flexgen.flex_opt $CMD

