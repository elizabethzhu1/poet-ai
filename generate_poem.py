import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data processing

# read csv file
df = pd.read_csv('poetry.csv')
df = df.head(50)
print(df)

# fill nan values with ''
df['Tags'].fillna(value='', inplace=True)

poems_series = df['Poem']
poems = poems_series.values

tokenizer = tf.keras.preprocessing.text.Tokenizer()

# extracting features and labels

# step 1 - tokenize
combined_poems = ' '.join(poems)  # join poems
combined_poems = combined_poems.replace('\r', '')

# set to other txt file
# with open('top-poems.txt', 'r') as file:
#     combined_poems = file.read()

# clean joined poems + split new lines
corpus = combined_poems.lower().split("\n")
corpus = [x for x in corpus if x != '']  # filter out empty sentences

# gets word index based on corpus
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # add one to offset 1-index

# step 2 - convert to numerical embeddings
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]  # converts line to array of ints
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# "let us go then you and I" -> input_sequences = ["let", "let us", "let us go", "let us go then", ...]

# step 3 - padding lines - creates same-length vector embeddings for each line

# get length of longest line
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.utils.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# separate data into inputs and labels
xs = input_sequences[:,:-1]  # select all rows/cols except last one
labels = input_sequences[:,-1]  # select all rows and only last col

# one hot encoding
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# train neural network -- pass sequence -> gives next value
def train():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(total_words, 250, input_length=max_sequence_len - 1))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(custom_learning_rate)
    adam = tf.keras.optimizers.legacy.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(xs, ys, epochs=30, callbacks=[lr_scheduler], verbose=1)
    model.save('poetry-foundation.keras')

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def custom_learning_rate(epoch):
    if epoch < 10:
        return 0.01
    elif epoch >= 10 and epoch < 15:
        return 0.005
    elif epoch >= 15 and epoch < 25:
        return 0.0001
    elif epoch >= 25 and epoch < 30:
        return 0.00001


print(tokenizer.word_index)

# generate poem: pass sequence -> generate word -> pass updated sequence -> generate next word
def generate_poem():
    model = tf.keras.models.load_model("poetry-foundation.keras")
    model.summary()  # show loaded model's architecture-
    seed = input("Enter a phrase to start a poem: ")
    next_words = 36

    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = tf.keras.utils.pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)

        prediction = np.argmax(predicted, axis=1)  # get word with highest probability
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == prediction:
                output_word = word
                break
        
        if (i % 6 == 0):
            seed += "\n"  # add line break every 6 lines (for now)
            seed += output_word
        else:
            seed += " " + output_word

    print("your generated poem: \n")
    print(seed)
    print('\n')

train()
generate_poem()

# ADDITIONAL TINKERING BELOW

# samples index from probabiltiy distribution (array)
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)

#     probas = np.random.multinomial(1, preds[0], 1)
#     return np.argmax(probas)

# can also weigh nltk similar words more based on the tag

# filter poems based on specific tags

# methods for finetuning poetry ai based on specific tags 
# 1. skew the weights for words in a poem that have a 'desired_tag'
# ^ avoid skewing weights too much
# 3. reinforcement learning (over the top) -- gradient
# 4. finetune (i.e. further train) on poems that contain desired tags -- if they don't, get nltk similar tags -- freeze pre-train layers + train more layers + compile + fit with specific poem dataset

# desired_tags = seed.split()

# for desired_tag in desired_tags:
#     tagged_poems = df[df['Tags'] == desired_tag]

    # make the weights of the words in that poem greater


# initialize dict mapping tag -> array of poems
# tag_to_poem = {}

# for index, row in df.iterrows():
#     title = row['Title']
#     poem = row['Poem']
#     tag = row['Tags']

#     # clean tag
#     if tag != '':
#         no_commas_tag = tag.replace(',', ' ')  # replace comma with space
#         cleaned_tag = no_commas_tag.replace('&', ' ')  # replace ampersand with space

#         # array of tags
#         tags = cleaned_tag.split()
        
#         for tag in tags:
#             lower_tag = tag.lower()
#             if lower_tag in tag_to_poem:
#                 tag_to_poem[lower_tag].append(poem)
#             else:
#                 tag_to_poem[lower_tag] = []
#                 tag_to_poem[lower_tag].append(poem)
