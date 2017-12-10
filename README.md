# Chung Generator 鍾子偉產生器

An LSTM word generator built with tensorflow as backend.  
在tensorflow上用LSTM做成的文字產生器  

I wrote an article on it on Medium. Check it out [here](https://medium.com/@vina.wt.chang/beb930dad45a)  
*我在Medium上寫了一篇說明文章，你可以在[這裡](https://medium.com/@vina.wt.chang/beb930dad45a)看*  

## Prerequisites
* tensorflow
* numpy

If you want to run the crawler yourself:  
*如果你想從頭跑爬蟲的話，你也需要:*  
* scrapy

## Just check how the pretrained generator works 只想玩玩看鍾子偉產生器

If you just want to play with the pretrained generator. After cloning the repo, run  
*如果你只是想玩玩看我train好的鍾子偉產生器，在clone之後*  

```
python generate.py
```

## Train a generator with your own data 想用自己的資料訓練產生器

If you want to try training your own generator with your own data, first update hyperparameters in config.py, then update the following parameters in train.py  
*如果你想用自己的資料訓練產生器, 先更新config.py裡的hyperparameter，再更新train.py裡這些參數*  

* word_to_id: dict mapping word to word_id
* words: list of words
* training: list of word_ids
* validation: list of word_ids

And run:  
*然後下*  
```
python train.py
```

Lastly, update "model_name" in generate.py. If you didn't modify the name in train.py, then model_name should be changed to "model".  
*最後更新generate.py裡的model_name，如果你沒修改train.py裡的model_name的話，那就把model_name改成"model"*  

You can now test out your generator by running:  
*用以下指令開啟產生器來玩玩看*  

```
python generate.py
```
