from pyabsa import available_checkpoints
from sentiment_classifier import SentimentClassifier
import time

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints(show_ckpts=True)

classifier = SentimentClassifier('english',
                                     auto_device='cuda:6',  # False means load model on CPU
                                     cal_perplexity=True,
                                     )

# # instance inference
# classifier.predict(['I love this movie, it is so great!'],
#                    save_result=True,
#                    print_result=True,  # print the result
#                    ignore_error=True,  # ignore the error when the model cannot predict the input
#                    )

# inference_source = APC.APCDatasetList.Restaurant16
start_time = time.time()
# target_file='113.laptop14/Laptops_Test_Gold.xml.seg.inference'  #
# target_file='115.restaurant15/restaurant_test.raw.inference'  #
# target_file='114.restaurant14/Restaurants_Test_Gold.xml.seg.inference'
# target_file='116.restaurant16/restaurant_test.raw.inference'
# target_file='101.ACL_Twitter/acl-14-short-data/test.raw.inference'
target_file='109.MAMS/test.xml.dat.inference'
apc_result, process_time = classifier.batch_predict(target_file=target_file,  #
                                      save_result=False,
                                      print_result=False,  # print the result
                                      pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                      eval_batch_size=64
                                      )
end_time = time.time()
print('Inference time (plus prepocessing): ', end_time - start_time)
print('Inference time: ', process_time)
# print(apc_result)