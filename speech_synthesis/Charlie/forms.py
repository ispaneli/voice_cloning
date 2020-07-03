from django import forms
from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = User
        fields = ('username', 'email', 'password')


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('website', 'picture')


class SynthesizerForm(forms.ModelForm):
    """
        Форма для синтезатора с опциями выбора готовых образцов
    """
    choices = [('1', 'Durak(male voice)'), ('2', 'Moriak(male voice)'),
               ('3', 'Bee(new gender)'), ('4', 'Your Voice')]
    choices = forms.ChoiceField(choices=choices, label="Voice Sample")

    class Meta:
        model = Synthesizer
        fields = ('text', 'input_audio')