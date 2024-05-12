## 我用的指令

- train
Markup :  `python3 train.py --dataroot ./datasets/ROAD_pix2pix/train --name ROAD_pix2pix --model pix2pix --direction BtoA --gpu_ids 1 --preprocess resize --epoch 10 --dataset_mode unaligned --serial_batches --load_size 512 --no_flip --input_nc 4
`

- test
Markup :  `python3 test.py --dataroot ./datasets/ROAD_pix2pix_test --name ROAD_pix2pix --model test --direction BtoA --gpu_ids 1 --preprocess resize --load_size 512 --dataset_mode single --norm batch --netG unet_256 --num_test 720 --input_nc 4
`
