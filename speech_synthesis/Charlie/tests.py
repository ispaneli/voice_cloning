from django.test import TestCase
from django.contrib.auth.models import User
from Charlie.models import UserProfile, Synthesizer
from Charlie.config import MICROSERVER_IP
from Charlie.tasks import send_file
from pathlib import Path
from django.urls import reverse
from speech_synthesis.celery import app
import os
import requests


class RegistrationTest(TestCase):
    """
        Тестируем создание пользователей и корректное заполнение всех полей
    """

    def setUp(self) -> None:
        self.users = len(User.objects.all())
        self.user = User.objects.create_user('sample', 'sample@mail.com', '12345')

        self.profile = UserProfile.objects.create(user=self.user)
        self.profile.user = self.user
        self.profile.picture = 'populate/дурак.jpg'
        self.profile.website = 'charlie.com'
        self.profile.save()

    def test_registration(self):
        self.assertEqual(self.users + 1, len(User.objects.all()),
                         "Amount of users didn't change")
        self.assertEqual(self.user.username, 'sample', 'Incorrect Username')
        self.assertEqual(self.user.email, 'sample@mail.com', 'Incorrect email')
        self.assertEqual(self.user.check_password("12345"), True, 'Incorrect password')

    def test_profile(self):
        self.assertEqual(self.profile.user, self.user, "Incorrect User")
        self.assertEqual(self.profile.website, 'charlie.com', "Incorrect Website")
        self.assertEqual(self.profile.picture, 'populate/дурак.jpg', "Incorrect Picture")


class S2SInteraction(TestCase):
    """
        Тестируем взаимодействие двух серверов
    """

    def test_connection(self):
        self.assertEqual(requests.get(MICROSERVER_IP).status_code, requests.codes.ok,
                         "No connection to the main model.")

    def test_interaction(self):
        self.assertEqual(send_file(MICROSERVER_IP, 'hello', 'media/recorded_sound.wav', 'test'),
                         "OK", "No connection")
        result = Path('media/recorded_soundtest.wav')
        self.assertEqual(result.exists(), True, "File has not been received")
        self.assertEqual(os.path.getsize(result) > 0, True, "Empty file")
        with (self.assertRaises(ValueError)):
            send_file(MICROSERVER_IP, '12345', 'media/recorded_sound.wav', 'test')
        os.remove('media/recorded_soundtest.wav')


class DjangoInnerLogic(TestCase):
    """
        Тесты на внутренние ограничения нашего сайта
    """
    def setUp(self) -> None:
        user = User.objects.get_or_create(username='testuser',
                                          first_name='Test',
                                          last_name='User',
                                          email='test@test.com')[0]
        user.set_password('testabc123')
        user.save()


    def test_access(self):
        response = self.client.get(reverse('Charlie:synthesizer'))
        self.assertEqual(response.status_code, 302, "We can access without login")

        self.client.login(username='testuser', password='testabc123')
        response = self.client.get(reverse('Charlie:synthesizer'))
        self.assertTrue(response.status_code, 200)
