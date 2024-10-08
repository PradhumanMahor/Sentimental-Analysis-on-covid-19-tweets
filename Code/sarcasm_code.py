import nltk
# import vader

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def remove_sarcasm(text):
    # initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # tokenize the text
    tokens = nltk.word_tokenize(text)
    # print(tokens)
    # tag the tokens with their parts of speech
    pos_tags = nltk.pos_tag(tokens)

    # identify the sentiment of each sentence
    sentiment_scores = []
    for sentence in nltk.sent_tokenize(text):
        score = sid.polarity_scores(sentence)
        sentiment_scores.append(score['compound'])

    # identify the most sarcastic sentences
    sarcastic_sentences = []
    for i, score in enumerate(sentiment_scores):
        if score < -0.5:
            sarcastic_sentences.append(nltk.sent_tokenize(text)[i])

    # remove the sarcastic sentences
    for sentence in sarcastic_sentences:
        text = text.replace(sentence, '')

    return text

# test the sarcasm removal function
text = "I just love getting stuck in traffic for hours on end. It's the highlight of my day."
print("Original text:", text)
text = remove_sarcasm(text)
print("Text with sarcasm removed:", text)
