from json import load
import difflib
import pymorphy2
import Levenshtein
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords = stopwords.words("russian")


def clean_string(s):
    s = ''.join([word for word in s if word not in punctuation]).lower()
    s = ' '.join([word for word in s.split() if word not in stopwords])
    return s


def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()


with open('checking.json') as f:
    terms = load(f)

v_a = {"К эмпатии": 0,
       "Общение с людьми": 0,
       "К работе с худ. образами": 0,
       "К работе с условными знаками": 0,
       "К здравой оцнеке своих действий": 0,
       "К усвоеннию материала": 0,
       "К работе с данными": 0,
       "К анализу данных": 0,
       "К работе с техникой": 0,
       "К расчётам": 0}

v_b = {"К эмпатии": 0,
       "Общение с людьми": 0,
       "К работе с художественными образами образами": 0,
       "К работе с условными знаками": 0,
       "К здравой оцнеке своих действий": 0,
       "К усвоеннию материала": 0,
       "К работе с данными": 0,
       "К анализу данных": 0,
       "К работе с техникой": 0,
       "К расчётам": 0}

v_c = {"Базовые знания HTML и CSS": 0,
       "Работа с графами": 0,
       "Программирование на GO": 0,
       "Программирование на JS": 0,
       "Программирование на Rust": 0,
       "Программирование на Python": 0,
       "Программирование на C++": 0,
       "Базовые знания работы с БД": 0,
       "Работа с сетью": 0}

vectors = [v_a, v_b, v_c]
morph = pymorphy2.MorphAnalyzer()

params_for_a_1 = {"знать": {"языки": {"К работе с данными": .5,
                                      "Общение с людьми": 0.5,
                                      "К работе с худ. образами": 0.3,
                                      "К усвоеннию материала": 1},
                            "python": {
                                "К анализу данных": .5,
                                "К работе с данными": .5},
                            "go": {"Общение с людьми": 0.3},
                            "javascript": {
                                "Общение с людьми": 0.3,
                                "К работе с худ. образами": 0.3},
                            "js": {
                                "Общение с людьми": 0.3,
                                "К работе с худ. образами": 0.3},
                            "c++": {
                                "К работе с техникой": 0.7,
                                "К усвоеннию материала": 0.3,
                                "К здравой оцнеке своих действий": 0.3},
                            "cpp": {
                                "К работе с техникой": 0.7,
                                "К усвоеннию материала": 0.3,
                                "К здравой оцнеке своих действий": 0.3},
                            "c": {
                                "К работе с техникой": 0.7,
                                "К усвоеннию материала": 0.3,
                                "К здравой оцнеке своих действий": 0.2}},
                  "думать": {},
                  "чувствовать": {},
                  "любить": {"писать"},
                  "обожат": {},
                  "предпочитать": {},
                  "ценить": {},
                  "следовать": {},
                  "оценивать": {},
                  "сопоставлять": {},
                  "выявлять": {"закономерность": {},
                               "ошибка": {"К здравой оцнеке своих действий": 0.8,
                                          "К расчётам": 0.3,
                                          "К анализу данных": 0.7}, "": {}}}
params_for_b_1 = {"решать": {},
                  "писать": {},
                  "усваивать": {},
                  "анализировать": {},
                  "работать": {},
                  "оценивать": {},
                  "программировать": {},"расчитывать": {}, "считать": {}}

params_for_c_1 = {"верстать": {
                     "сайт": {"Базовые знания HTML и CSS": 1,
                              "Работа с сетью": .3}},
                  "писать": {
                     "python": {"Программирование на Python": 1}},
                  "владеть": {
                      "python": {"Программирование на Python": 1},
                      "go": {"Программирование на GO": 1}},
                  "анализировать": {
                      "таблица": {"Базовые знания работы с БД": 1},
                      "график": {"Работа с графами": 1},
                      "код": {"Программирование на GO": .2,
                              "Программирование на JS": .2,
                              "Программирование на Rust": .2,
                              "Программирование на C++": 0.2,
                              }},
                  "работать": {
                      "код": {"Программирование на GO": 0.5,
                              "Программирование на JS": 0.5,
                              "Программирование на Rust": 0.5,
                              "Программирование на C++": 0.5,
                              },
                      "график": {"Работа с графами": 1},
                      "таблица": {"Базовые знания работы с БД": 1}},
                  "строить": {
                      "график": {"Работа с графами": 0.3},
                      "таблица": {"Базовые знания работы с БД": 0.5}}}

# params_for_c_2 = {"сайт": {"Базовые знания HTML и CSS", "Работа с сетью"}}

get_params = dict()


for i_terms in terms:
    for param in terms[i_terms]:
        f = i_terms.lower()
        x = list(clean_string(param).split(" "))
        for w in x:
            w = max(morph.parse(w), key=lambda z: z.score)
            if "INFN" == w.tag.POS:
                f = w.normal_form
                break
        k = []

        for w in x:
            w = max(morph.parse(w), key=lambda z: z.score)
            # if "NOUN" == w.tag.POS or "ADJF" == w.tag.POS:
            if "INFN" != w.tag.POS:
                # print(w)
                k.append(w.normal_form)
        if f not in get_params:
            get_params.setdefault(f, k)
        else:
            get_params[f] += k


# for i in get_params:
#     print(i)
#     print("--------")
#     print(*get_params[i])
#     # for k in get_params[i]:
#     #     print(k)
#     print("=======")


def anal(v, p_m):
    for i_p in get_params:
        for k_p in p_m:
            if Levenshtein.ratio(i_p, k_p) > .7:
                for pa in get_params[i_p]:
                    for ideal in p_m[k_p]:
                        # print(Levenshtein.ratio(pa, ideal), pa, ideal)
                        if Levenshtein.ratio(pa, ideal) > .7:
                            for lvl in p_m[k_p][ideal]:
                                v[lvl] += p_m[k_p][ideal][lvl]


# for i_p in get_params:
#     # print(i_p)
#     # print('----------')
#     for k_p in params_for_c_1:
#         if Levenshtein.ratio(i_p, k_p) > .7:
#             for param in get_params[i_p]:
#                 for ideal in params_for_c_1[k_p]:
#                     print(Levenshtein.ratio(param, ideal), param, ideal)
#                     if Levenshtein.ratio(param, ideal) > .7:
#                         for l in params_for_c_1[k_p][ideal]:
#                             v_c[l] += params_for_c_1[k_p][ideal][l]


anal(v_c, params_for_c_1)
print("===========")
for i in v_c:
    print(i, v_c[i])
anal(v_a, params_for_a_1)
print("===========")
for i in v_a:
    print(i, v_a[i])

# print("===========")
# for i in v_c:
#     print(i, v_c[i])