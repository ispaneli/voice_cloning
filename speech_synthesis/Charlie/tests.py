from django.test import TestCase
from django.contrib.auth.models import User
from Charlie.models import UserProfile, Synthesizer



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
