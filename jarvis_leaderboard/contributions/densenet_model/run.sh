
pip install atomvision
# https://figshare.com/articles/figure/AtomVision_data/16788268 
train_classifier_cnn.py  --model vgg --train_folder /wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/train_folder --test_folder /wrk/knc6/AtomVision/Combined/J2D_C2D_2DMatP/test_folder --epochs 50 --batch_size 16
