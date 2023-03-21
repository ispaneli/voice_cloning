import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE',
                      'speech_synthesis.settings')

import django
django.setup()

from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User
from django.db import IntegrityError


class PopulateDatabase:
    """Класс для заполнения БД пользователями и информацией о них"""

    # модели для заполнения БД
    ADD_MODELS = {'profile': UserProfile,
                  'synthesizer': Synthesizer}

    def __init__(self, user=None):
        self.user = user


    def get_or_create(self, username, email, password):
        """ Создает или возвращает пользователя для заполнения БД"""
        try:
            self.user = User.objects.create_user(username, email, password)
        except IntegrityError:
            self.user = User.objects.get(username=username, email=email)


    def add(self, method: str, **kwargs):
        """ Добавляем модель и заполняем её поля значениями """
        model = self.ADD_MODELS[method].objects.get_or_create(user=self.user)[0]
        for key, value in kwargs.items():
            model.__dict__[key] = value
        model.save()


def populate():
    users = [{'username': 'durak',
              'email': 'durak@mail.ru',
              'password': 'durak'},
             {'username': 'moriak',
              'email': 'moriak@mail.ru',
              'password': 'moriak'},
             {'username': 'pchela',
              'email': 'pchela@mail.ru',
              'password': 'pchela'},
             ]

    sample_profiles = [{'website': 'charlie.com',
                        'picture': 'populate/дурак.jpg'},
                       {'website': 'charlie.com',
                        'picture': 'populate/moriak.jpg'},
                       {'website': 'charlie.com',
                        'picture': 'populate/bee.jpg'},
                       ]

    voice_samples = ({'export_audio': 'voice_samples/first.mp3'},
                     {'export_audio': 'voice_samples/second.mp3'},
                     {'export_audio': 'voice_samples/third.mp3'})

    for user, sample, voice in zip(users, sample_profiles, voice_samples):
        pop = PopulateDatabase()
        pop.get_or_create(user['username'], user['email'], user['password'])
        pop.add('profile', **sample)
        pop.add('synthesizer', **voice)


if __name__ == '__main__':
    print("Starting Django population script...")
    populate()
