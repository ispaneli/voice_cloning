from django.shortcuts import render, redirect, reverse
from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User
from Charlie.forms import UserForm, UserProfileForm, SynthesizerForm
from django.contrib.auth.decorators import login_required
from django.views import View
from django.utils.decorators import method_decorator
import requests
import struct
import wave

class Index(View):
    """
        Класс, отвечающий за главную страницу
    """
    @staticmethod
    def get(request):
        return render(request, 'Charlie/index.html')


class AboutView(View):
    """
        Класс, отвечающий за страницу информации
    """
    @staticmethod
    def get(request):
        return render(request, 'Charlie/about.html')


class RegisterProfileView(View):
    """
        Класс, отвечающий за страницу настройки дополнительной информации профиля,
        появляющейся сразу после создания профиля

        :picture:   добавляется картинка пользователя
        :website:   добавляется вебстраница пользователя
    """
    @method_decorator(login_required)
    def get(self, request):
        form = UserProfileForm()
        return render(request, 'Charlie/profile_registration.html',
                      context={'form': form})

    @method_decorator(login_required)
    def post(self, request):
        form = UserProfileForm(request.POST, request.FILES)
        if form.is_valid():
            user_profile = form.save(commit=False)
            user_profile.user = request.user
            user_profile.save()

            return redirect(reverse('Charlie:index'))
        else:
            print(form.errors)


def post_helper(form, user):
    """
        Вспомогательная функция для сохранения формы, исключает
        дублирование кода в классах ProfileView и SynthesizerView

        :param form: форма для заполнения
        :param user: текущий пользователь
        :return: None
    """
    if form.is_valid():
        form.save(commit=True)
        return redirect('Charlie:profile', user.username)
    else:
        print(form.errors)


class Details:
    """
        Класс для предоставления деталей о пользователе. Возвращает пользователя,
        нужную модель и форму для заполнения

        :method_dict: нужен для выбора необходимой формы в наших методах класса
    """
    method_dict = {'user': (UserProfile, UserProfileForm),
                   'synthesizer': (Synthesizer, SynthesizerForm)}

    @classmethod
    def provide_details(cls, username, parameter):
        """
            Интерфейс предоставления деталей пользователя
        :param username: пользователь
        :param parameter: нужный метод
        :return: user, our_model, form
        """
        method = cls.method_dict[parameter]
        try:
            user, our_model, form = cls.get_user_details(username, method)
        except TypeError:
            return redirect(reverse('Charlie:index'))

        return user, our_model, form

    @staticmethod
    def get_user_details(username, model):
        """
            Реализация интерфейса предоставления деталей
        :param username: пользователь
        :param model: метод
        :return: return user, our_model, form
        """
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return None

        our_model = model[0].objects.get_or_create(user=user)[0]
        form = model[1](model[0].filling_params(our_model))

        return user, our_model, form


class ProfileView(View):
    """
        Класс настройки и просмотра нашего профиля
    """

    @method_decorator(login_required)
    def get(self, request, username):
        user, user_profile, form = Details.provide_details(username, 'user')
        synthesizer = Synthesizer.objects.get_or_create(user=user)[0]
        context_dict = {'user_profile': user_profile,
                        'selected_user': user,
                        'form': form,
                        'synthesizer': synthesizer}

        return render(request, 'Charlie/profile.html', context_dict)

    @method_decorator(login_required)
    def post(self, request, username):
        user, user_profile, form = Details.provide_details(username, 'user')
        form = UserProfileForm(request.POST, request.FILES, instance=user_profile)
        post_helper(form, user)
        context_dict = {'user_profile': user_profile,
                        'selected_user': user,
                        'form': form}

        return render(request, 'Charlie/profile.html', context_dict)


class ListProfilesView(View):
    """
        Класс предоставляет список всех пользователей
    """
    @method_decorator(login_required)
    def get(self, request):
        profiles = UserProfile.objects.all()

        return render(request, 'Charlie/list_profiles.html',
                      context={'user_profile_list': profiles})


class SynthesizerView(View):
    """
        Класс обращается к синтезатору и возвращает произведенную запись
    """
    path = ''

    @method_decorator(login_required)
    def get(self, request):

        user, synthesizer, form = Details.provide_details(request.user, 'synthesizer')
        context_dict = {'synthesizer': synthesizer,
                        'form': form,
                        }
        return render(request, 'Charlie/synthesizer.html', context=context_dict)

    @method_decorator(login_required)
    def post(self, request):

        user, synthesizer, form = Details.provide_details(request.user, 'synthesizer')
        form = SynthesizerForm(request.POST, request.FILES, instance=synthesizer)
        post_helper(form, user)
        context_dict = {'synthesizer': synthesizer,
                        'form': form,
                        }

        path = "media/" + str(synthesizer.input_audio)
        audio = {'text': request.POST['text'],
                 'audio': open(path, 'rb')}

        a = requests.post('http://192.168.1.98:5005',
                          data={'text': request.POST['text']},
                          files=audio)

        f = wave.open('media/recorded_sound.wav', 'w')
        f.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        f.writeframes(a._content)
        f.close()

        synthesizer.export_audio = 'recorded_sound.wav'
        return render(request, 'Charlie/synthesizer.html', context=context_dict)


class SampleView(View):
    """
        После заселения БД доступна страничка образцов. Из БД берутся три
        профиля по умолчанию.
    """
    def get(self, request):
        users = ('durak', 'moriak', 'pchela')
        context = {}
        for user in users:
            det_user, user_profile, _ = Details.provide_details(user, 'user')
            context[user] = user_profile
            synthesizer = Synthesizer.objects.get_or_create(user=det_user)[0]
            context[user + "_synth"] = synthesizer
        return render(request, 'Charlie/sample.html', context=context)
