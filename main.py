import numpy as np
import string
import os
import re
import unicodedata

def search_dir(dir):
    dir_content = {}
    for i, (r, d, f) in enumerate(os.walk(dir)):
        for sub_d in d:
            if sub_d not in dir_content:
                dir_content[sub_d] = []
            for file in os.listdir(os.path.join(r, sub_d)):
                if file.endswith(".txt"):
                    with open(os.path.join(r, sub_d, file), 'r', encoding='utf-8') as file:
                        content = file.read()
                        dir_content[sub_d].append(content)
    return dir_content

def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-zA-Z]', '', text)
    text = text.lower()
    return text

def create_vector(text):
    text = preprocess_text(text)
    letter_count = {char: 0 for char in string.ascii_lowercase}
    total_letters = 0

    for char in text:
        if char.isalpha():
            if char in letter_count:
                letter_count[char] += 1
                total_letters += 1

    letter_proportions = [count / total_letters for count in letter_count.values()]
    return letter_proportions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(texts, labels, epochs=1100, learning_rate=0.05):
    for epoch in range(epochs):
        for text, label in zip(texts, labels):
            vector = np.array(create_vector(text))
            for lang in languages:
                expected = 1 if lang == label else 0
                output = sigmoid(np.dot(vector, weights[lang]))
                error = expected - output
                weights[lang] += learning_rate * error * vector

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def classify(text):
    vector = np.array(create_vector(text))
    outputs = {lang: np.dot(vector, weights[lang]) for lang in languages}
    probabilities = softmax(np.array(list(outputs.values())))
    predicted_lang_index = np.argmax(probabilities)
    predicted_lang = languages[predicted_lang_index]
    return predicted_lang

train_data_path = '/Users/oluniaxoxo/Desktop/NAI/Jednowarstwowa siec neuronowa/train_data'
dir_content = search_dir(train_data_path)

languages = ['Czech', 'Slovakian', 'Spanish']
weights = {lang: np.random.rand(26) for lang in languages}

texts = []
labels = []

for label, text_list in dir_content.items():
    texts.extend(text_list)
    labels.extend([label] * len(text_list))

train(texts, labels)

def test(dir):
    dir_content = {}
    for i, (r, d, f) in enumerate(os.walk(dir)):
        for sub_d in d:
            if sub_d not in dir_content:
                dir_content[sub_d] = []
            for file in os.listdir(os.path.join(r, sub_d)):
                if file.endswith(".txt"):
                    with open(os.path.join(r, sub_d, file), 'r', encoding='utf-8') as file:
                        content = file.read()
                        dir_content[sub_d].append(content)

    all = 0
    correct = 0
    for key, value in dir_content.items():
        for text in value:
            output = classify(text)
            print("Expected: " + key + ", output: " + output)
            all += 1
            if(key == output): correct +=1
    accuracy = (correct / all) * 100
    print("Accuracy: " + str(accuracy))

test_data_path = "/Users/oluniaxoxo/Desktop/NAI/Jednowarstwowa siec neuronowa/test_data"
test(test_data_path)
