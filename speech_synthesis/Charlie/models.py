from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class UserProfile(models.Model):
    """
        Модель профиля пользователя.Пользователь имеет картинку и свой сайт
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    website = models.URLField(blank=True)
    picture = models.ImageField(upload_to='profile_images', blank=True)

    @staticmethod
    def filling_params(user_profile):
        """
        :param user_profile: профиль пользователя
        :return: заполненный значениями пользователя словарь
        """
        return {'website': user_profile.website,
                'picture': user_profile.picture}

    def __str__(self):
        return self.user.username


class Synthesizer(models.Model):
    """
        Модель синтезатора.Принимает на вход образец голоса, текст для озвучивания.
        На выход модель отдает озвученный текст
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    text = models.TextField(blank=True)
    input_audio = models.FileField(upload_to='voice_samples', blank=True)
    export_audio = models.FileField(upload_to='machine_voice', blank=True)

    @staticmethod
    def filling_params(synthesizer):
        """
        :param synthesizer: синтезатор
        :return: заполненный значениями синтезатора словарь
        """
        return {'text': synthesizer.text,
                'input_audio': synthesizer.input_audio,
                }
