def load_data(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
data = load_data("got.txt")


def clean_text(text):
    sample = text
    sample = re.sub('[%s]' % re.escape(string.punctuation), '', sample)
    sample = [word for word in sample.split() if word.isalpha()]
    sample = [word.lower() for word in sample]
    sample = " ".join(sample)

    return sample

cleaned_data = clean_text(data)

print(cleaned_data[:200])
print('Total Tokens: %d' % len(cleaned_data))
print('Unique Tokens: %d' % len(set(cleaned_data)))

print('Total Tokens: %d' % len(cleaned_data.split()))
print('Unique Tokens: %d' % len(set(cleaned_data.split())))

sequences_doc = []
seq_len = 50
l = seq_len + 1
tokens = [w for w in cleaned_data.split()]

for i in range(l, len(tokens)):
    seq = tokens[i - l:i]

    line = ' '.join(seq)
    sequences_doc.append(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences_doc)
sequences = tokenizer.texts_to_sequences(sequences_doc)

vocab_size = len(tokenizer.word_index) + 1

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

seq_length = X.shape[1]


def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    return model

model = define_model(vocab_size, seq_length)

model.fit(X, y, batch_size=128, epochs=10)

model.save('text_gen_model.h5')
# save the tokenizer
pickle.dump(tokenizer, open('tokenizer_text_gen.pkl', 'wb'))


###Generate text

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
    # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        yhat = np.argmax(yhat,axis=1)
        print(yhat)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

seed_text = sequences_doc[np.random.randint(0,len(sequences_doc))]
print(seed_text + '\n')
generate_seq(model, tokenizer, seq_length, seed_text, 50)


seed_text = sequences_doc[np.random.randint(0,len(sequences_doc))]
print(seed_text + '\n')
generate_seq(model, tokenizer, seq_length, seed_text, 50)[:60]

pd.read_csv("twitter_parsed_dataset.csv")["Text"][np.random.randint(0, len(pd.read_csv("twitter_parsed_dataset.csv")))]

seed_text = sequences_doc[np.random.randint(0,len(sequences_doc))]
print(seed_text + '\n')
generate_seq(model, tokenizer, seq_length, seed_text, 50)[:60]





