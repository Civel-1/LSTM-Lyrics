from __future__ import print_function
from keras.layers import LSTM, Bidirectional, Embedding, Dense, Dropout, Activation
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.models import Sequential
import numpy as np
from matplotlib import pyplot as plt
import io

# PATH PARAMETERS
CORPUS_PATH = "kanye_west.txt"
OUTPUT_PATH = "examples.txt"

# DATA PARAMETERS
MIN_WORD_FREQUENCY = 2
SEQUENCE_LEN = 2

# EMBEDDING PARAMETERS
OUTPUTDIM = 1024

# LSTM PARAMETERS
DROPOUT = 0.2
OUTPUT_SPACE = 128
ACTIVATION_FUNCTION = 'softmax'

# TRAINING PARAMETERS
BATCH_SIZE = 32
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
NUMBER_EPOCH = 30

# RESULT PARAMETERS
RESULT_LEN = 50
DIVERSITY = [0.3, 0.4, 0.5, 0.6, 0.7]

# variables globales
indices_to_words = dict()
words_to_indices = dict()


def get_data():
    # on ouvre le fichier avec le corpus de paroles et on effectue une petite refacto pour que le retour à la ligne
    # soit considéré comme un mot à part entier
    with io.open(CORPUS_PATH, encoding='utf-8') as f:
        text = f.read().lower().replace('\n', ' \n ')
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace(",", "")
        text = text.replace("...", " ... ")
    print('Données récupérées ! Longueur du corpus en charactères :', len(text))
    return text


# on prépare la données, on sépare le texte en mots, ignore ceux qui ne sont pas redondants
def prepare_data():
    text_in_words = [w for w in raw_text.split(' ') if w.strip() != '' or w == '\n']
    print('Nombre de mots dans le corpus :', len(text_in_words))

    # on calcule la fréquence de chaque mot
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # on ignore tous les mots dont la fréquence est inférieure à celle en paramètre
    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    words = set(text_in_words)
    print('Mots unique avant sélection :', len(words))
    print('Mots ignorés si fréquence <', MIN_WORD_FREQUENCY)
    # petit trick permettant de faire la différence de deux sets
    words = sorted(set(words) - ignored_words)
    print('Mots unique après sélection :', len(words))

    # On découpe maintenant le texte complet en phrases semi-redondantes égales à la longueur SEQUENCE_LEN
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Maintenant on retire les sequences qui comportent au moins un mot présent de la liste des mots à
        # ignorer
        if len(set(text_in_words[i: i + SEQUENCE_LEN + 1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Séquences ignorées :', ignored)
    print('Séquences restantes :', len(sentences))

    # on crée des dictionnaires qui permettent d'associer chaque mot à un entier ( et un second pour chaque entier à
    # un mot ). Cela permet de manipuler des données moins lourdes et donc gagner en temps de traitement.
    # ils seront utilisés par le model lors de l'entraînement et des tests
    global words_to_indices
    words_to_indices = dict((c, i) for i, c in enumerate(words))
    global indices_to_words
    indices_to_words = dict((i, c) for i, c in enumerate(words))

    return words, sentences, next_words


# percentage-test signifie que que 2% du corpus de phrases est retiré de la liste et utilisé comme données de test
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # mélange des phrases
    print('Mélange des phrases')
    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    # découpage en 2 sous listes : entraînement et test selon le partage 98-2%
    cut_index = int(len(sentences_original) * (1. - (percentage_test / 100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Taille des données d'entraînement = %d" % len(x_train))
    print("Taille des données de test = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def create_model():
    print('Création du model')
    global model
    model = Sequential()
    # le Dropout est une technique qui permet d'ignorer aléatoirement certains neurones durant l'entraînement.
    # d'embedding qui permet d'ajouter du contexte
    model.add(Embedding(input_dim=len(words), output_dim=OUTPUTDIM))
    # création du réseau lstm bidirectionnel
    model.add(Bidirectional(LSTM(OUTPUT_SPACE)))
    if DROPOUT > 0:
        model.add(Dropout(DROPOUT))
    model.add(Dense(len(words)))
    # on ajoute une couche d'activation densément connecté au layer précédent
    model.add(Activation(ACTIVATION_FUNCTION))


# Function d'entraînement de notre modèle. C'est un apprentissage supervisé donc on fournit également des données de
# test
def train_model():
    model.compile(loss=LOSS_FUNCTION, optimizer="adam", metrics=['accuracy'])

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20)
    callbacks_list = [print_callback, early_stopping]

    history = model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                               steps_per_epoch=int(len(sentences) / BATCH_SIZE) + 1,
                               epochs=NUMBER_EPOCH,
                               callbacks=callbacks_list,
                               validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                               validation_steps=int(len(sentences_test) / BATCH_SIZE) + 1)
    print_result(history)

# Fonction utilisée lors de l'entraînement du model. Permet de générer des données à entraîner et évaluer
# les generator permettent d'itérer sur une liste une seule fois, ils itèrent puis oublie ce qu'ils avaient en mémoire
# sans pour autant repasser dessus. Cela permet d'éviter des problèmes de ressources. Le batch size définit le nombre de
# phrases données par itération.
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        # instancie puis remplit les matrices des données
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = words_to_indices[w]
            y[i] = words_to_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        # retourne des générateurs, x(phrase) et y(next word)
        yield x, y


# function appelée à chaque fin d'epoch afin d'écrire des échantillons de résultats.
def on_epoch_end(epoch, logs):
    output_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # On prend une phrase choisie aléatoirement dans tout le corpus comme graine.
    seed_index = np.random.randint(len(sentences + sentences_test))
    seed = (sentences + sentences_test)[seed_index]

    # on crée un échantillon pour chaque valeur dans le tableau des diversités
    for diversity in DIVERSITY:
        sentence = seed
        output_file.write('----- Diversity:' + str(diversity) + '\n')
        output_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        output_file.write(' '.join(sentence))

        # on crée ensuite l'échantillon grâce à la fonction de keras selon le paramètre
        for i in range(RESULT_LEN):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sentence):
                x_pred[0, t] = words_to_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_to_words[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            output_file.write(" " + next_word)
        output_file.write('\n')
    output_file.write('=' * 80 + '\n')
    output_file.flush()


# Function issue de keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def print_result(history):
    loss_values = history.history['loss']
    acc_values = history.history['acc']
    epochs = range(1, len(loss_values) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc_values, label='Training Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# ---------------- MAIN --------------------

# récupération des données
raw_text = get_data()
# préparation des données pour le réseau de neurones
words, sentences, next_words = prepare_data()
# Préparation des données d'entrainement et de test
(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)
# création du réseau neuronal
create_model()
# entraînement et création d'échantillons
output_file = open(OUTPUT_PATH, "w")
train_model()

