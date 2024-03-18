---
language:
- en
tags:
- text-classification
- zero-shot-classification
pipeline_tag: zero-shot-classification
library_name: transformers
license: mit
---

# Model description:  deberta-v3-large-zeroshot-v1.1-all-33
The model is designed for zero-shot classification with the Hugging Face pipeline. 

The model can do one universal classification task: determine whether a hypothesis is "true" or "not true" given a text
(`entailment` vs. `not_entailment`).  
This task format is based on the Natural Language Inference task (NLI).
The task is so universal that any classification task can be reformulated into this task.

A detailed description of how the model was trained and how it can be used is available in this [paper](https://arxiv.org/pdf/2312.17543.pdf).

## Training data
The model was trained on a mixture of __33 datasets and 387 classes__ that have been reformatted into this universal format.
1.   Five NLI datasets with ~885k texts: "mnli", "anli", "fever", "wanli", "ling"
2.   28 classification tasks reformatted into the universal NLI format. ~51k cleaned texts were used to avoid overfitting:
'amazonpolarity', 'imdb', 'appreviews', 'yelpreviews', 'rottentomatoes',
'emotiondair', 'emocontext', 'empathetic',
'financialphrasebank', 'banking77', 'massive',
'wikitoxic_toxicaggregated', 'wikitoxic_obscene', 'wikitoxic_threat', 'wikitoxic_insult', 'wikitoxic_identityhate', 
'hateoffensive', 'hatexplain', 'biasframes_offensive', 'biasframes_sex', 'biasframes_intent',
'agnews', 'yahootopics',
'trueteacher', 'spam', 'wellformedquery',
'manifesto', 'capsotu'.

See details on each dataset here: https://github.com/MoritzLaurer/zeroshot-classifier/blob/main/datasets_overview.csv

Note that compared to other NLI models, this model predicts two classes (`entailment` vs. `not_entailment`)
as opposed to three classes (entailment/neutral/contradiction)

The model was only trained on English data. For __multilingual use-cases__, 
I recommend machine translating texts to English with libraries like [EasyNMT](https://github.com/UKPLab/EasyNMT).
English-only models tend to perform better than multilingual models and
validation with English data can be easier if you don't speak all languages in your corpus.

### How to use the model
#### Simple zero-shot classification pipeline
```python
#!pip install transformers[sentencepiece]
from transformers import pipeline
text = "Angela Merkel is a politician in Germany and leader of the CDU"
hypothesis_template = "This example is about {}"
classes_verbalized = ["politics", "economy", "entertainment", "environment"]
zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
output = zeroshot_classifier(text, classes_verbalised, hypothesis_template=hypothesis_template, multi_label=False)
print(output)
```

### Details on data and training
The code for preparing the data and training & evaluating the model is fully open-source here: https://github.com/MoritzLaurer/zeroshot-classifier/tree/main

Hyperparameters and other details are available in this Weights & Biases repo: https://wandb.ai/moritzlaurer/deberta-v3-large-zeroshot-v1-1-all-33/table?workspace=user-


## Metrics 

Balanced accuracy is reported for all datasets. 
`deberta-v3-large-zeroshot-v1.1-all-33` was trained on all datasets, with only maximum 500 texts per class to avoid overfitting. 
The metrics on these datasets are therefore not strictly zeroshot, as the model has seen some data for each task during training. 
`deberta-v3-large-zeroshot-v1.1-heldout` indicates zeroshot performance on the respective dataset.
To calculate these zeroshot metrics, the pipeline was run 28 times, each time with one dataset held out from training to simulate a zeroshot setup.

![figure_large_v1.1](https://raw.githubusercontent.com/MoritzLaurer/zeroshot-classifier/main/results/fig_large_v1.1.png)


|                            |   deberta-v3-large-mnli-fever-anli-ling-wanli-binary |   deberta-v3-large-zeroshot-v1.1-heldout |   deberta-v3-large-zeroshot-v1.1-all-33 |
|:---------------------------|----------------------------:|-----------------------------------------:|----------------------------------------:|
| datasets mean (w/o nli)    |                        64.1 |                                     73.4 |                                    85.2 |
| amazonpolarity (2)         |                        94.7 |                                     96.6 |                                    96.8 |
| imdb (2)                   |                        90.3 |                                     95.2 |                                    95.5 |
| appreviews (2)             |                        93.6 |                                     94.3 |                                    94.7 |
| yelpreviews (2)            |                        98.5 |                                     98.4 |                                    98.9 |
| rottentomatoes (2)         |                        83.9 |                                     90.5 |                                    90.8 |
| emotiondair (6)            |                        49.2 |                                     42.1 |                                    72.1 |
| emocontext (4)             |                        57   |                                     69.3 |                                    82.4 |
| empathetic (32)            |                        42   |                                     34.4 |                                    58   |
| financialphrasebank (3)    |                        77.4 |                                     77.5 |                                    91.9 |
| banking77 (72)             |                        29.1 |                                     52.8 |                                    72.2 |
| massive (59)               |                        47.3 |                                     64.7 |                                    77.3 |
| wikitoxic_toxicaggreg (2)  |                        81.6 |                                     86.6 |                                    91   |
| wikitoxic_obscene (2)      |                        85.9 |                                     91.9 |                                    93.1 |
| wikitoxic_threat (2)       |                        77.9 |                                     93.7 |                                    97.6 |
| wikitoxic_insult (2)       |                        77.8 |                                     91.1 |                                    92.3 |
| wikitoxic_identityhate (2) |                        86.4 |                                     89.8 |                                    95.7 |
| hateoffensive (3)          |                        62.8 |                                     66.5 |                                    88.4 |
| hatexplain (3)             |                        46.9 |                                     61   |                                    76.9 |
| biasframes_offensive (2)   |                        62.5 |                                     86.6 |                                    89   |
| biasframes_sex (2)         |                        87.6 |                                     89.6 |                                    92.6 |
| biasframes_intent (2)      |                        54.8 |                                     88.6 |                                    89.9 |
| agnews (4)                 |                        81.9 |                                     82.8 |                                    90.9 |
| yahootopics (10)           |                        37.7 |                                     65.6 |                                    74.3 |
| trueteacher (2)            |                        51.2 |                                     54.9 |                                    86.6 |
| spam (2)                   |                        52.6 |                                     51.8 |                                    97.1 |
| wellformedquery (2)        |                        49.9 |                                     40.4 |                                    82.7 |
| manifesto (56)             |                        10.6 |                                     29.4 |                                    44.1 |
| capsotu (21)               |                        23.2 |                                     69.4 |                                    74   |
| mnli_m (2)                 |                        93.1 |                                    nan   |                                    93.1 |
| mnli_mm (2)                |                        93.2 |                                    nan   |                                    93.2 |
| fevernli (2)               |                        89.3 |                                    nan   |                                    89.5 |
| anli_r1 (2)                |                        87.9 |                                    nan   |                                    87.3 |
| anli_r2 (2)                |                        76.3 |                                    nan   |                                    78   |
| anli_r3 (2)                |                        73.6 |                                    nan   |                                    74.1 |
| wanli (2)                  |                        82.8 |                                    nan   |                                    82.7 |
| lingnli (2)                |                        90.2 |                                    nan   |                                    89.6 |



## Limitations and bias
The model can only do text classification tasks. 

Please consult the original DeBERTa paper and the papers for the different datasets for potential biases. 


## License
The base model (DeBERTa-v3) is published under the MIT license.
The datasets the model was fine-tuned on are published under a diverse set of licenses.
The following table provides an overview of the non-NLI datasets used for fine-tuning, 
information on licenses, the underlying papers etc.: https://github.com/MoritzLaurer/zeroshot-classifier/blob/main/datasets_overview.csv

## Citation
If you use this model academically, please cite: 
```
@misc{laurer_building_2023,
	title = {Building {Efficient} {Universal} {Classifiers} with {Natural} {Language} {Inference}},
	url = {http://arxiv.org/abs/2312.17543},
	doi = {10.48550/arXiv.2312.17543},
	abstract = {Generative Large Language Models (LLMs) have become the mainstream choice for fewshot and zeroshot learning thanks to the universality of text generation. Many users, however, do not need the broad capabilities of generative LLMs when they only want to automate a classification task. Smaller BERT-like models can also learn universal tasks, which allow them to do any text classification task without requiring fine-tuning (zeroshot classification) or to learn new tasks with only a few examples (fewshot), while being significantly more efficient than generative LLMs. This paper (1) explains how Natural Language Inference (NLI) can be used as a universal classification task that follows similar principles as instruction fine-tuning of generative LLMs, (2) provides a step-by-step guide with reusable Jupyter notebooks for building a universal classifier, and (3) shares the resulting universal classifier that is trained on 33 datasets with 389 diverse classes. Parts of the code we share has been used to train our older zeroshot classifiers that have been downloaded more than 55 million times via the Hugging Face Hub as of December 2023. Our new classifier improves zeroshot performance by 9.4\%.},
	urldate = {2024-01-05},
	publisher = {arXiv},
	author = {Laurer, Moritz and van Atteveldt, Wouter and Casas, Andreu and Welbers, Kasper},
	month = dec,
	year = {2023},
	note = {arXiv:2312.17543 [cs]},
	keywords = {Computer Science - Artificial Intelligence, Computer Science - Computation and Language},
}
```

### Ideas for cooperation or questions?
If you have questions or ideas for cooperation, contact me at m{dot}laurer{at}vu{dot}nl or [LinkedIn](https://www.linkedin.com/in/moritz-laurer/)

### Debugging and issues
Note that DeBERTa-v3 was released on 06.12.21 and older versions of HF Transformers can have issues running the model (e.g. resulting in an issue with the tokenizer). Using Transformers>=4.13 might solve some issues.

### Hypotheses used for classification
The hypotheses in the tables below were used to fine-tune the model. 
Inspecting them can help users get a feeling for which type of hypotheses and tasks the model was trained on.
You can formulate your own hypotheses by changing the `hypothesis_template` of the zeroshot pipeline. For example:

```python
from transformers import pipeline
text = "Angela Merkel is a politician in Germany and leader of the CDU"
hypothesis_template = "Merkel is the leader of the party: {}"
classes_verbalized = ["CDU", "SPD", "Greens"]
zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33")
output = zeroshot_classifier(text, classes_verbalised, hypothesis_template=hypothesis_template, multi_label=False)
print(output)
```

Note that a few rows in the `massive` and `banking77` datasets contain `nan` because some classes were so ambiguous/unclear that I excluded them from the data. 


#### wellformedquery
| label           | hypothesis                                     |
|:----------------|:-----------------------------------------------|
| not_well_formed | This example is not a well formed Google query |
| well_formed     | This example is a well formed Google query.    |
#### biasframes_sex
| label   | hypothesis                                                 |
|:--------|:-----------------------------------------------------------|
| not_sex | This example does not contain allusions to sexual content. |
| sex     | This example contains allusions to sexual content.         |
#### biasframes_intent
| label      | hypothesis                                                       |
|:-----------|:-----------------------------------------------------------------|
| intent     | The intent of this example is to be offensive/disrespectful.     |
| not_intent | The intent of this example is not to be offensive/disrespectful. |
#### biasframes_offensive
| label         | hypothesis                                                               |
|:--------------|:-------------------------------------------------------------------------|
| not_offensive | This example could not be considered offensive, disrespectful, or toxic. |
| offensive     | This example could be considered offensive, disrespectful, or toxic.     |
#### financialphrasebank
| label    | hypothesis                                                                |
|:---------|:--------------------------------------------------------------------------|
| negative | The sentiment in this example is negative from an investor's perspective. |
| neutral  | The sentiment in this example is neutral from an investor's perspective.  |
| positive | The sentiment in this example is positive from an investor's perspective. |
#### rottentomatoes
| label    | hypothesis                                                             |
|:---------|:-----------------------------------------------------------------------|
| negative | The sentiment in this example rotten tomatoes movie review is negative |
| positive | The sentiment in this example rotten tomatoes movie review is positive |
#### amazonpolarity
| label    | hypothesis                                                      |
|:---------|:----------------------------------------------------------------|
| negative | The sentiment in this example amazon product review is negative |
| positive | The sentiment in this example amazon product review is positive |
#### imdb
| label    | hypothesis                                                  |
|:---------|:------------------------------------------------------------|
| negative | The sentiment in this example imdb movie review is negative |
| positive | The sentiment in this example imdb movie review is positive |
#### appreviews
| label    | hypothesis                                            |
|:---------|:------------------------------------------------------|
| negative | The sentiment in this example app review is negative. |
| positive | The sentiment in this example app review is positive. |
#### yelpreviews
| label    | hypothesis                                             |
|:---------|:-------------------------------------------------------|
| negative | The sentiment in this example yelp review is negative. |
| positive | The sentiment in this example yelp review is positive. |
#### wikitoxic_toxicaggregated
| label               | hypothesis                                                      |
|:--------------------|:----------------------------------------------------------------|
| not_toxicaggregated | This example wikipedia comment does not contain toxic language. |
| toxicaggregated     | This example wikipedia comment contains toxic language.         |
#### wikitoxic_obscene
| label       | hypothesis                                                        |
|:------------|:------------------------------------------------------------------|
| not_obscene | This example wikipedia comment does not contain obscene language. |
| obscene     | This example wikipedia comment contains obscene language.         |
#### wikitoxic_threat
| label      | hypothesis                                                |
|:-----------|:----------------------------------------------------------|
| not_threat | This example wikipedia comment does not contain a threat. |
| threat     | This example wikipedia comment contains a threat.         |
#### wikitoxic_insult
| label      | hypothesis                                                 |
|:-----------|:-----------------------------------------------------------|
| insult     | This example wikipedia comment contains an insult.         |
| not_insult | This example wikipedia comment does not contain an insult. |
#### wikitoxic_identityhate
| label            | hypothesis                                                     |
|:-----------------|:---------------------------------------------------------------|
| identityhate     | This example wikipedia comment contains identity hate.         |
| not_identityhate | This example wikipedia comment does not contain identity hate. |
#### hateoffensive
| label       | hypothesis                                                              |
|:------------|:------------------------------------------------------------------------|
| hate_speech | This example tweet contains hate speech.                                |
| neither     | This example tweet contains neither offensive language nor hate speech. |
| offensive   | This example tweet contains offensive language without hate speech.     |
#### hatexplain
| label       | hypothesis                                                                                 |
|:------------|:-------------------------------------------------------------------------------------------|
| hate_speech | This example text from twitter or gab contains hate speech.                                |
| neither     | This example text from twitter or gab contains neither offensive language nor hate speech. |
| offensive   | This example text from twitter or gab contains offensive language without hate speech.     |
#### spam
| label    | hypothesis                    |
|:---------|:------------------------------|
| not_spam | This example sms is not spam. |
| spam     | This example sms is spam.     |
#### emotiondair
| label    | hypothesis                                         |
|:---------|:---------------------------------------------------|
| anger    | This example tweet expresses the emotion: anger    |
| fear     | This example tweet expresses the emotion: fear     |
| joy      | This example tweet expresses the emotion: joy      |
| love     | This example tweet expresses the emotion: love     |
| sadness  | This example tweet expresses the emotion: sadness  |
| surprise | This example tweet expresses the emotion: surprise |
#### emocontext
| label   | hypothesis                                                                            |
|:--------|:--------------------------------------------------------------------------------------|
| angry   | This example tweet expresses the emotion: anger                                       |
| happy   | This example tweet expresses the emotion: happiness                                   |
| others  | This example tweet does not express any of the emotions: anger, sadness, or happiness |
| sad     | This example tweet expresses the emotion: sadness                                     |
#### empathetic
| label        | hypothesis                                                 |
|:-------------|:-----------------------------------------------------------|
| afraid       | The main emotion of this example dialogue is: afraid       |
| angry        | The main emotion of this example dialogue is: angry        |
| annoyed      | The main emotion of this example dialogue is: annoyed      |
| anticipating | The main emotion of this example dialogue is: anticipating |
| anxious      | The main emotion of this example dialogue is: anxious      |
| apprehensive | The main emotion of this example dialogue is: apprehensive |
| ashamed      | The main emotion of this example dialogue is: ashamed      |
| caring       | The main emotion of this example dialogue is: caring       |
| confident    | The main emotion of this example dialogue is: confident    |
| content      | The main emotion of this example dialogue is: content      |
| devastated   | The main emotion of this example dialogue is: devastated   |
| disappointed | The main emotion of this example dialogue is: disappointed |
| disgusted    | The main emotion of this example dialogue is: disgusted    |
| embarrassed  | The main emotion of this example dialogue is: embarrassed  |
| excited      | The main emotion of this example dialogue is: excited      |
| faithful     | The main emotion of this example dialogue is: faithful     |
| furious      | The main emotion of this example dialogue is: furious      |
| grateful     | The main emotion of this example dialogue is: grateful     |
| guilty       | The main emotion of this example dialogue is: guilty       |
| hopeful      | The main emotion of this example dialogue is: hopeful      |
| impressed    | The main emotion of this example dialogue is: impressed    |
| jealous      | The main emotion of this example dialogue is: jealous      |
| joyful       | The main emotion of this example dialogue is: joyful       |
| lonely       | The main emotion of this example dialogue is: lonely       |
| nostalgic    | The main emotion of this example dialogue is: nostalgic    |
| prepared     | The main emotion of this example dialogue is: prepared     |
| proud        | The main emotion of this example dialogue is: proud        |
| sad          | The main emotion of this example dialogue is: sad          |
| sentimental  | The main emotion of this example dialogue is: sentimental  |
| surprised    | The main emotion of this example dialogue is: surprised    |
| terrified    | The main emotion of this example dialogue is: terrified    |
| trusting     | The main emotion of this example dialogue is: trusting     |
#### agnews
| label    | hypothesis                                             |
|:---------|:-------------------------------------------------------|
| Business | This example news text is about business news          |
| Sci/Tech | This example news text is about science and technology |
| Sports   | This example news text is about sports                 |
| World    | This example news text is about world news             |
#### yahootopics
| label                  | hypothesis                                                                                         |
|:-----------------------|:---------------------------------------------------------------------------------------------------|
| Business & Finance     | This example question from the Yahoo Q&A forum is categorized in the topic: Business & Finance     |
| Computers & Internet   | This example question from the Yahoo Q&A forum is categorized in the topic: Computers & Internet   |
| Education & Reference  | This example question from the Yahoo Q&A forum is categorized in the topic: Education & Reference  |
| Entertainment & Music  | This example question from the Yahoo Q&A forum is categorized in the topic: Entertainment & Music  |
| Family & Relationships | This example question from the Yahoo Q&A forum is categorized in the topic: Family & Relationships |
| Health                 | This example question from the Yahoo Q&A forum is categorized in the topic: Health                 |
| Politics & Government  | This example question from the Yahoo Q&A forum is categorized in the topic: Politics & Government  |
| Science & Mathematics  | This example question from the Yahoo Q&A forum is categorized in the topic: Science & Mathematics  |
| Society & Culture      | This example question from the Yahoo Q&A forum is categorized in the topic: Society & Culture      |
| Sports                 | This example question from the Yahoo Q&A forum is categorized in the topic: Sports                 |
#### massive
| label                    | hypothesis                                                                                |
|:-------------------------|:------------------------------------------------------------------------------------------|
| alarm_query              | The example utterance is a query about alarms.                                            |
| alarm_remove             | The intent of this example utterance is to remove an alarm.                               |
| alarm_set                | The intent of the example utterance is to set an alarm.                                   |
| audio_volume_down        | The intent of the example utterance is to lower the volume.                               |
| audio_volume_mute        | The intent of this example utterance is to mute the volume.                               |
| audio_volume_other       | The example utterance is related to audio volume.                                         |
| audio_volume_up          | The intent of this example utterance is turning the audio volume up.                      |
| calendar_query           | The example utterance is a query about a calendar.                                        |
| calendar_remove          | The intent of the example utterance is to remove something from a calendar.               |
| calendar_set             | The intent of this example utterance is to set something in a calendar.                   |
| cooking_query            | The example utterance is a query about cooking.                                           |
| cooking_recipe           | This example utterance is about cooking recipies.                                         |
| datetime_convert         | The example utterance is related to date time changes or conversion.                      |
| datetime_query           | The intent of this example utterance is a datetime query.                                 |
| email_addcontact         | The intent of this example utterance is adding an email address to contacts.              |
| email_query              | The example utterance is a query about emails.                                            |
| email_querycontact       | The intent of this example utterance is to query contact details.                         |
| email_sendemail          | The intent of the example utterance is to send an email.                                  |
| general_greet            | This example utterance is a general greet.                                                |
| general_joke             | The intent of the example utterance is to hear a joke.                                    |
| general_quirky           | nan                                                                                       |
| iot_cleaning             | The intent of the example utterance is for an IoT device to start cleaning.               |
| iot_coffee               | The intent of this example utterance is for an IoT device to make coffee.                 |
| iot_hue_lightchange      | The intent of this example utterance is changing the light.                               |
| iot_hue_lightdim         | The intent of the example utterance is to dim the lights.                                 |
| iot_hue_lightoff         | The example utterance is related to turning the lights off.                               |
| iot_hue_lighton          | The example utterance is related to turning the lights on.                                |
| iot_hue_lightup          | The intent of this example utterance is to brighten lights.                               |
| iot_wemo_off             | The intent of this example utterance is turning an IoT device off.                        |
| iot_wemo_on              | The intent of the example utterance is to turn an IoT device on.                          |
| lists_createoradd        | The example utterance is related to creating or adding to lists.                          |
| lists_query              | The example utterance is a query about a list.                                            |
| lists_remove             | The intent of this example utterance is to remove a list or remove something from a list. |
| music_dislikeness        | The intent of this example utterance is signalling music dislike.                         |
| music_likeness           | The example utterance is related to liking music.                                         |
| music_query              | The example utterance is a query about music.                                             |
| music_settings           | The intent of the example utterance is to change music settings.                          |
| news_query               | The example utterance is a query about the news.                                          |
| play_audiobook           | The example utterance is related to playing audiobooks.                                   |
| play_game                | The intent of this example utterance is to start playing a game.                          |
| play_music               | The intent of this example utterance is for an IoT device to play music.                  |
| play_podcasts            | The example utterance is related to playing podcasts.                                     |
| play_radio               | The intent of the example utterance is to play something on the radio.                    |
| qa_currency              | This example utteranceis about currencies.                                                |
| qa_definition            | The example utterance is a query about a definition.                                      |
| qa_factoid               | The example utterance is a factoid question.                                              |
| qa_maths                 | The example utterance is a question about maths.                                          |
| qa_stock                 | This example utterance is about stocks.                                                   |
| recommendation_events    | This example utterance is about event recommendations.                                    |
| recommendation_locations | The intent of this example utterance is receiving recommendations for good locations.     |
| recommendation_movies    | This example utterance is about movie recommendations.                                    |
| social_post              | The example utterance is about social media posts.                                        |
| social_query             | The example utterance is a query about a social network.                                  |
| takeaway_order           | The intent of this example utterance is to order takeaway food.                           |
| takeaway_query           | This example utterance is about takeaway food.                                            |
| transport_query          | The example utterance is a query about transport or travels.                              |
| transport_taxi           | The intent of this example utterance is to get a taxi.                                    |
| transport_ticket         | This example utterance is about transport tickets.                                        |
| transport_traffic        | This example utterance is about transport or traffic.                                     |
| weather_query            | This example utterance is a query about the wheather.                                     |
#### banking77
| label                                            | hypothesis                                                                                                |
|:-------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| Refund_not_showing_up                            | This customer example message is about a refund not showing up.                                           |
| activate_my_card                                 | This banking customer example message is about activating a card.                                         |
| age_limit                                        | This banking customer example message is related to age limits.                                           |
| apple_pay_or_google_pay                          | This banking customer example message is about apple pay or google pay                                    |
| atm_support                                      | This banking customer example message requests ATM support.                                               |
| automatic_top_up                                 | This banking customer example message is about automatic top up.                                          |
| balance_not_updated_after_bank_transfer          | This banking customer example message is about a balance not updated after a transfer.                    |
| balance_not_updated_after_cheque_or_cash_deposit | This banking customer example message is about a balance not updated after a cheque or cash deposit.      |
| beneficiary_not_allowed                          | This banking customer example message is related to a beneficiary not being allowed or a failed transfer. |
| cancel_transfer                                  | This banking customer example message is related to the cancellation of a transfer.                       |
| card_about_to_expire                             | This banking customer example message is related to the expiration of a card.                             |
| card_acceptance                                  | This banking customer example message is related to the scope of acceptance of a card.                    |
| card_arrival                                     | This banking customer example message is about the arrival of a card.                                     |
| card_delivery_estimate                           | This banking customer example message is about a card delivery estimate or timing.                        |
| card_linking                                     | nan                                                                                                       |
| card_not_working                                 | This banking customer example message is about a card not working.                                        |
| card_payment_fee_charged                         | This banking customer example message is about a card payment fee.                                        |
| card_payment_not_recognised                      | This banking customer example message is about a payment the customer does not recognise.                 |
| card_payment_wrong_exchange_rate                 | This banking customer example message is about a wrong exchange rate.                                     |
| card_swallowed                                   | This banking customer example message is about a card swallowed by a machine.                             |
| cash_withdrawal_charge                           | This banking customer example message is about a cash withdrawal charge.                                  |
| cash_withdrawal_not_recognised                   | This banking customer example message is about an unrecognised cash withdrawal.                           |
| change_pin                                       | This banking customer example message is about changing a pin code.                                       |
| compromised_card                                 | This banking customer example message is about a compromised card.                                        |
| contactless_not_working                          | This banking customer example message is about contactless not working                                    |
| country_support                                  | This banking customer example message is about country-specific support.                                  |
| declined_card_payment                            | This banking customer example message is about a declined card payment.                                   |
| declined_cash_withdrawal                         | This banking customer example message is about a declined cash withdrawal.                                |
| declined_transfer                                | This banking customer example message is about a declined transfer.                                       |
| direct_debit_payment_not_recognised              | This banking customer example message is about an unrecognised direct debit payment.                      |
| disposable_card_limits                           | This banking customer example message is about the limits of disposable cards.                            |
| edit_personal_details                            | This banking customer example message is about editing personal details.                                  |
| exchange_charge                                  | This banking customer example message is about exchange rate charges.                                     |
| exchange_rate                                    | This banking customer example message is about exchange rates.                                            |
| exchange_via_app                                 | nan                                                                                                       |
| extra_charge_on_statement                        | This banking customer example message is about an extra charge.                                           |
| failed_transfer                                  | This banking customer example message is about a failed transfer.                                         |
| fiat_currency_support                            | This banking customer example message is about fiat currency support                                      |
| get_disposable_virtual_card                      | This banking customer example message is about getting a disposable virtual card.                         |
| get_physical_card                                | nan                                                                                                       |
| getting_spare_card                               | This banking customer example message is about getting a spare card.                                      |
| getting_virtual_card                             | This banking customer example message is about getting a virtual card.                                    |
| lost_or_stolen_card                              | This banking customer example message is about a lost or stolen card.                                     |
| lost_or_stolen_phone                             | This banking customer example message is about a lost or stolen phone.                                    |
| order_physical_card                              | This banking customer example message is about ordering a card.                                           |
| passcode_forgotten                               | This banking customer example message is about a forgotten passcode.                                      |
| pending_card_payment                             | This banking customer example message is about a pending card payment.                                    |
| pending_cash_withdrawal                          | This banking customer example message is about a pending cash withdrawal.                                 |
| pending_top_up                                   | This banking customer example message is about a pending top up.                                          |
| pending_transfer                                 | This banking customer example message is about a pending transfer.                                        |
| pin_blocked                                      | This banking customer example message is about a blocked pin.                                             |
| receiving_money                                  | This banking customer example message is about receiving money.                                           |
| request_refund                                   | This banking customer example message is about a refund request.                                          |
| reverted_card_payment?                           | This banking customer example message is about reverting a card payment.                                  |
| supported_cards_and_currencies                   | nan                                                                                                       |
| terminate_account                                | This banking customer example message is about terminating an account.                                    |
| top_up_by_bank_transfer_charge                   | nan                                                                                                       |
| top_up_by_card_charge                            | This banking customer example message is about the charge for topping up by card.                         |
| top_up_by_cash_or_cheque                         | This banking customer example message is about topping up by cash or cheque.                              |
| top_up_failed                                    | This banking customer example message is about top up issues or failures.                                 |
| top_up_limits                                    | This banking customer example message is about top up limitations.                                        |
| top_up_reverted                                  | This banking customer example message is about issues with topping up.                                    |
| topping_up_by_card                               | This banking customer example message is about topping up by card.                                        |
| transaction_charged_twice                        | This banking customer example message is about a transaction charged twice.                               |
| transfer_fee_charged                             | This banking customer example message is about an issue with a transfer fee charge.                       |
| transfer_into_account                            | This banking customer example message is about transfers into the customer's own account.                 |
| transfer_not_received_by_recipient               | This banking customer example message is about a transfer that has not arrived yet.                       |
| transfer_timing                                  | This banking customer example message is about transfer timing.                                           |
| unable_to_verify_identity                        | This banking customer example message is about an issue with identity verification.                       |
| verify_my_identity                               | This banking customer example message is about identity verification.                                     |
| verify_source_of_funds                           | This banking customer example message is about the source of funds.                                       |
| verify_top_up                                    | This banking customer example message is about verification and top ups                                   |
| virtual_card_not_working                         | This banking customer example message is about a virtual card not working                                 |
| visa_or_mastercard                               | This banking customer example message is about types of bank cards.                                       |
| why_verify_identity                              | This banking customer example message questions why identity verification is necessary.                   |
| wrong_amount_of_cash_received                    | This banking customer example message is about a wrong amount of cash received.                           |
| wrong_exchange_rate_for_cash_withdrawal          | This banking customer example message is about a wrong exchange rate for a cash withdrawal.               |
#### trueteacher
| label                  | hypothesis                                                           |
|:-----------------------|:---------------------------------------------------------------------|
| factually_consistent   | The example summary is factually consistent with the full article.   |
| factually_inconsistent | The example summary is factually inconsistent with the full article. |
#### capsotu
| label                 | hypothesis                                                                                                |
|:----------------------|:----------------------------------------------------------------------------------------------------------|
| Agriculture           | This example text from a US presidential speech is about agriculture                                      |
| Civil Rights          | This example text from a US presidential speech is about civil rights or minorities or civil liberties    |
| Culture               | This example text from a US presidential speech is about cultural policy                                  |
| Defense               | This example text from a US presidential speech is about defense or military                              |
| Domestic Commerce     | This example text from a US presidential speech is about banking or finance or commerce                   |
| Education             | This example text from a US presidential speech is about education                                        |
| Energy                | This example text from a US presidential speech is about energy or electricity or fossil fuels            |
| Environment           | This example text from a US presidential speech is about the environment or water or waste or pollution   |
| Foreign Trade         | This example text from a US presidential speech is about foreign trade                                    |
| Government Operations | This example text from a US presidential speech is about government operations or administration          |
| Health                | This example text from a US presidential speech is about health                                           |
| Housing               | This example text from a US presidential speech is about community development or housing issues          |
| Immigration           | This example text from a US presidential speech is about migration                                        |
| International Affairs | This example text from a US presidential speech is about international affairs or foreign aid             |
| Labor                 | This example text from a US presidential speech is about employment or labour                             |
| Law and Crime         | This example text from a US presidential speech is about law, crime or family issues                      |
| Macroeconomics        | This example text from a US presidential speech is about macroeconomics                                   |
| Public Lands          | This example text from a US presidential speech is about public lands or water management                 |
| Social Welfare        | This example text from a US presidential speech is about social welfare                                   |
| Technology            | This example text from a US presidential speech is about space or science or technology or communications |
| Transportation        | This example text from a US presidential speech is about transportation                                   |
#### manifesto
| label                                      | hypothesis                                                                                                                                                                                                              |
|:-------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agriculture and Farmers: Positive          | This example text from a political party manifesto is positive towards policies for agriculture and farmers                                                                                                             |
| Anti-Growth Economy: Positive              | This example text from a political party manifesto is in favour of anti-growth politics                                                                                                                                 |
| Anti-Imperialism                           | This example text from a political party manifesto is anti-imperialistic, for example against controlling other countries and for greater self-government of colonies                                                   |
| Centralisation                             | This example text from a political party manifesto is in favour of political centralisation                                                                                                                             |
| Civic Mindedness: Positive                 | This example text from a political party manifesto is positive towards national solidarity, civil society or appeals for public spiritedness or against anti-social attitudes                                           |
| Constitutionalism: Negative                | This example text from a political party manifesto is positive towards constitutionalism                                                                                                                                |
| Constitutionalism: Positive                | This example text from a political party manifesto is positive towards constitutionalism and the status quo of the constitution                                                                                         |
| Controlled Economy                         | This example text from a political party manifesto is supportive of direct government control of the economy, e.g. price control or minimum wages                                                                       |
| Corporatism/Mixed Economy                  | This example text from a political party manifesto is positive towards cooperation of government, employers, and trade unions simultaneously                                                                            |
| Culture: Positive                          | This example text from a political party manifesto is in favour of cultural policies or leisure facilities, for example museus, libraries or public sport clubs                                                         |
| Decentralization                           | This example text from a political party manifesto is for decentralisation or federalism                                                                                                                                |
| Democracy                                  | This example text from a political party manifesto favourably mentions democracy or democratic procedures or institutions                                                                                               |
| Economic Goals                             | This example text from a political party manifesto is a broad/general statement on economic goals without specifics                                                                                                     |
| Economic Growth: Positive                  | This example text from a political party manifesto is supportive of economic growth, for example facilitation of more production or government aid for growth                                                           |
| Economic Orthodoxy                         | This example text from a political party manifesto is for economic orthodoxy, for example reduction of budget deficits, thrift or a strong currency                                                                     |
| Economic Planning                          | This example text from a political party manifesto is positive towards government economic planning, e.g. policy plans or strategies                                                                                    |
| Education Expansion                        | This example text from a political party manifesto is about the need to expand/improve policy on education                                                                                                              |
| Education Limitation                       | This example text from a political party manifesto is sceptical towards state expenditure on education, for example in favour of study fees or private schools                                                          |
| Environmental Protection                   | This example text from a political party manifesto is in favour of environmental protection, e.g. fighting climate change or 'green' policies or preservation of natural resources or animal rights                     |
| Equality: Positive                         | This example text from a political party manifesto is positive towards equality or social justice, e.g. protection of underprivileged groups or fair distribution of resources                                          |
| European Community/Union: Negative         | This example text from a political party manifesto negatively mentions the EU or European Community                                                                                                                     |
| European Community/Union: Positive         | This example text from a political party manifesto is positive towards the EU or European Community, for example EU expansion and integration                                                                           |
| Foreign Special Relationships: Negative    | This example text from a political party manifesto is negative towards particular countries                                                                                                                             |
| Foreign Special Relationships: Positive    | This example text from a political party manifesto is positive towards particular countries                                                                                                                             |
| Free Market Economy                        | This example text from a political party manifesto is in favour of a free market economy and capitalism                                                                                                                 |
| Freedom and Human Rights                   | This example text from a political party manifesto is in favour of freedom and human rights, for example freedom of speech, assembly or against state coercion or for individualism                                     |
| Governmental and Administrative Efficiency | This example text from a political party manifesto is in favour of efficiency in government/administration, for example by restructuring civil service or improving bureaucracy                                         |
| Incentives: Positive                       | This example text from a political party manifesto is favourable towards supply side economic policies supporting businesses, for example for incentives like subsidies or tax breaks                                   |
| Internationalism: Negative                 | This example text from a political party manifesto is sceptical of internationalism, for example negative towards international cooperation, in favour of national sovereignty and unilaterialism                       |
| Internationalism: Positive                 | This example text from a political party manifesto is in favour of international cooperation with other countries, for example mentions the need for aid to developing countries, or global governance                  |
| Keynesian Demand Management                | This example text from a political party manifesto is for keynesian demand management and demand side economic policies                                                                                                 |
| Labour Groups: Negative                    | This example text from a political party manifesto is negative towards labour groups and unions                                                                                                                         |
| Labour Groups: Positive                    | This example text from a political party manifesto is positive towards labour groups, for example for good working conditions, fair wages or unions                                                                     |
| Law and Order: Positive                    | This example text from a political party manifesto is positive towards law and order and strict law enforcement                                                                                                         |
| Market Regulation                          | This example text from a political party manifesto is supports market regulation for a fair and open market, for example for consumer protection or for increased competition or for social market economy              |
| Marxist Analysis                           | This example text from a political party manifesto is positive towards Marxist-Leninist ideas or uses specific Marxist terminology                                                                                      |
| Middle Class and Professional Groups       | This example text from a political party manifesto favourably references the middle class, e.g. white colar groups or the service sector                                                                                |
| Military: Negative                         | This example text from a political party manifesto is negative towards the military, for example for decreasing military spending or disarmament                                                                        |
| Military: Positive                         | This example text from a political party manifesto is positive towards the military, for example for military spending or rearmament or military treaty obligations                                                     |
| Multiculturalism: Negative                 | This example text from a political party manifesto is sceptical towards multiculturalism, or for cultural integration or appeals to cultural homogeneity in society                                                     |
| Multiculturalism: Positive                 | This example text from a political party manifesto favourably mentions cultural diversity, for example for freedom of religion or linguistic heritages                                                                  |
| National Way of Life: Negative             | This example text from a political party manifesto unfavourably mentions a country's nation and history, for example sceptical towards patriotism or national pride                                                     |
| National Way of Life: Positive             | This example text from a political party manifesto is positive towards the national way of life and history, for example pride of citizenship or appeals to patriotism                                                  |
| Nationalisation                            | This example text from a political party manifesto is positive towards government ownership of industries or land or for economic nationalisation                                                                       |
| Non-economic Demographic Groups            | This example text from a political party manifesto favourably mentions non-economic demographic groups like women, students or specific age groups                                                                      |
| Peace                                      | This example text from a political party manifesto is positive towards peace and peaceful means of solving crises, for example in favour of negotiations and ending wars                                                |
| Political Authority                        | This example text from a political party manifesto mentions the speaker's competence to govern or other party's lack of such competence, or favourably mentions a strong/stable government                              |
| Political Corruption                       | This example text from a political party manifesto is negative towards political corruption or abuse of political/bureaucratic power                                                                                    |
| Protectionism: Negative                    | This example text from a political party manifesto is negative towards protectionism, in favour of free trade                                                                                                           |
| Protectionism: Positive                    | This example text from a political party manifesto is in favour of protectionism, for example tariffs, export subsidies                                                                                                 |
| Technology and Infrastructure: Positive    | This example text from a political party manifesto is about technology and infrastructure, e.g. the importance of modernisation of industry, or supportive of public spending on infrastructure/tech                    |
| Traditional Morality: Negative             | This example text from a political party manifesto is negative towards traditional morality, for example against religious moral values, for divorce or abortion, for modern families or separation of church and state |
| Traditional Morality: Positive             | This example text from a political party manifesto is favourable towards traditional or religious values, for example for censorship of immoral behavour, for traditional family values or religious institutions       |
| Underprivileged Minority Groups            | This example text from a political party manifesto favourably mentions underprivileged minorities, for example handicapped, homosexuals or immigrants                                                                   |
| Welfare State Expansion                    | This example text from a political party manifesto is positive towards the welfare state, e.g. health care, pensions or social housing                                                                                  |
| Welfare State Limitation                   | This example text from a political party manifesto is for limiting the welfare state, for example public funding for social services or social security, e.g. private care before state care                            |