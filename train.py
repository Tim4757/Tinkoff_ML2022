import re
import codecs
import numpy as np
from tqdm import tqdm
import pickle

class NgramModel(object):

    def __init__(self, n, path):
        self.n = n
        self.fragment = {} # словарь, который хранит список слов-кандидатов с учетом контекста
        self.ngram_counter = {} # отслеживает, сколько раз ngram появлялся в тексте раньше
        self.path = path

    def get_ngrams(self, tokens: list) -> list:
        n = self.n
        tokens = (n - 1) * ['<START>'] + tokens
        res = []
        for i in range(n - 1, len(tokens)):                      # объединение последовательных токенов в N-граммы
            temp = tuple([tokens[i - p - 1] for p in reversed(range(n - 1))])
            res.append((temp, tokens[i]))
        return res

    def update(self, sentence: str) -> None:
        ngrams = self.get_ngrams(sentence.split()) # токенизация

        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0
            # обновляем словари
            prev_words, target_word = ngram
            if prev_words in self.fragment:
                self.fragment[prev_words].append(target_word)
            else:
                self.fragment[prev_words] = [target_word]

    def prob(self, context, token):
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.fragment[context]))       # cчитается вероятность встретить token при данном контексте
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result

    def random_token(self, context):
        map_to_probs = {}

        try:
            token_of_interest = self.fragment[context]          # если такого фрагмента нет, решил ставить в данном случае точку
        except KeyError:
            return '.'

        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)   # считаем вероятность для каждого token

        return np.random.choice(list(map_to_probs.keys()))

    def generate(self,token_count: int):
        n = self.n

        context_queue = (n - 1) * ['<START>']
        result = []

        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '.':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)

    def fit(self):
        f = codecs.open(self.path, "r", "utf-8")
        text = f.read()
        text = text.split('.')             # разбиваем текст на предложения
        for sentence in tqdm(text):
            s = re.sub('[\W_]+', ' ', sentence.lower(), flags=re.UNICODE) #очищаем текст от лишних символом и приводим к нижнему регистру
            self.update(s)
        return self


m = NgramModel(4,'4_toma.txt')
m.fit()
pickle.dump(m, open('model.pkl', 'wb'))
