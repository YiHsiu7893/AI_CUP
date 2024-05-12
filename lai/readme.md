## 我用的指令

- train (因為目前發現沒有serial_batches跟no_flip的話A跟B的配對順序會錯誤，所以目前就先讓他依序訓練+不做資料增強，我這幾天再來改改看這部分)

  python3 train.py --dataroot ./datasets/ROAD_pix2pix/train --name ROAD_pix2pix --model pix2pix --direction BtoA --gpu_ids 1 --preprocess resize --epoch 10 --dataset_mode unaligned --serial_batches --load_size 512 --no_flip --input_nc 4

- test

  python3 test.py --dataroot ./datasets/ROAD_pix2pix_test --name ROAD_pix2pix --model test --direction BtoA --gpu_ids 1 --preprocess resize --load_size 512 --dataset_mode single --norm batch --netG unet_256 --num_test 720 --input_nc 4
