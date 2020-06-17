import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE',
                      'speech_synthesis.settings')

import django
django.setup()

from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User
from django.db import IntegrityError


def get_or_create(username, email, password):
    try:
        new_user = User.objects.create_user(username, email, password)
    except IntegrityError:
        return User.objects.get(username=username, email=email)

    return new_user


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

    voice_samples = ('voice_samples/first.mp3', 'voice_samples/second.mp3',
                     'voice_samples/third.mp3')

    for user, sample, voice in zip(users, sample_profiles, voice_samples):
        us_prof = get_or_create(user['username'], user['email'], user['password'])
        add_profile(us_prof, **sample)
        add_synthesizer(us_prof, voice)


def add_profile(user, website, picture):
    profile = UserProfile.objects.get_or_create(user=user)[0]
    profile.user = user
    profile.website = website
    profile.picture = picture
    profile.save()
    return profile


def add_synthesizer(user, export_audio):
    synthesizer = Synthesizer.objects.get_or_create(user=user)[0]
    synthesizer.export_audio = export_audio
    synthesizer.save()
    return synthesizer


if __name__ == '__main__':
    print("Starting Django population script...")
    populate()
