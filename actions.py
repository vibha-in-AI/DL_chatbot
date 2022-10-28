# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.models import Model




class ActionProdDetails(Action):

    def name(self) -> Text:
        return "action_prod_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prod_name = tracker.get_slot("prod_name")
        prod_feature = tracker.get_slot("prod_feature")


        print(f"You have queries regarding {prod_name} with {prod_feature}")
        
        dispatcher.utter_message(text="")

        return []

class ActionAskMore(Action):

    def name(self) -> Text:
        return "action_ask_more"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="How can I help you further?")

        return []

class ActionGetSentiment(Action):

    def name(self) -> Text:
        return "action_get_sentiment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = tracker.get_slot('feedback_text')

        print("data : {}".format(data))
        classifier = Classifier_Reviews()

        sentiment_message = str(classifier.get_sentiment(data))

        dispatcher.utter_message(text= sentiment_message)

        return []
    
class Classifier_Reviews:
    """Load in classifier & encoders"""

    def __init__(self):
        super(Classifier_Reviews, self).__init__()


        self.model = self.create_model()


    def create_model(self):

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        # bert layer
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

        preprocessed_text = bert_preprocess(text_input)
        output = bert_layer(preprocessed_text)

        output = tf.keras.layers.Dense(1, "sigmoid")(output['pooled_output'])
        # Bert model
        # We are using only pooled output not sequence out.
        # If you want to know about those, please read https://www.kaggle.com/questions-and-answers/86510
        model = Model(inputs=text_input, outputs=output)

        model.load_weights("sentiment_analyzer.h5")

        return model

    def get_sentiment(self, data):
    
        """Classify reviews"""

        # encode input data
        encoded_data = [data]

        pred = self.model.predict(encoded_data)

        print(pred)

        if pred > 0.7:
            message="Great! Thanks for the awesome review!"
        else:
            message = "Uhoh! Let's make it better for you! Our executive will call you shortly."
            
        return message
