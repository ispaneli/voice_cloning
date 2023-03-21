from django.shortcuts import render, redirect, reverse
from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User
from Charlie.forms import UserForm, UserProfileForm, SynthesizerForm
from django.contrib.auth.decorators import login_required
from django.views import View
from django.utils.decorators import method_decorator
import requests
from Charlie.config import MICROSERVER_IP
from Charlie.tasks import send_file
from speech_synthesis.celery import app
import time


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
        print(synthesizer.__dict__)
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
    choices = {'1': 'pre_voices/durak.wav', '2': 'pre_voices/moriak.wav',
               '3': 'pre_voices/durak.wav', '4': None}

    @method_decorator(login_required)
    def get(self, request):

        user, synthesizer, form = Details.provide_details(request.user, 'synthesizer')
        context_dict = {'synthesizer': synthesizer,
                        'form': form,
                        }
        return render(request, 'Charlie/synthesizer.html', context=context_dict)

    @method_decorator(login_required)
    def post(self, request):
        path = "media/"
        user, synthesizer, form = Details.provide_details(request.user, 'synthesizer')
        form = SynthesizerForm(request.POST, request.FILES, instance=synthesizer)

        if self.choices[request.POST['choices']] is not None:
            path += self.choices[request.POST['choices']]
        else:
            synthesizer.input_audio = request.FILES['input_audio']
            path += 'voice_samples/' + str(synthesizer.input_audio)

        post_helper(form, user)
        send = send_file.delay(MICROSERVER_IP, request.POST['text'], path, user.username)
        request.session['task_id'] = send.task_id
        return redirect('Charlie:loader')


class SampleView(View):
    """
        Отображение странички с статичными данными
    """
    def get(self, request):
        return render(request, 'Charlie/sample.html')


class LoaderView(View):
    """
        Страничка, появляющаяся после отправки формы
    """
    @method_decorator(login_required)
    def get(self, request):
        time.sleep(2)
        res = app.AsyncResult(request.session['task_id'])
        if res.state == 'FAILURE':
            return render(request, 'Charlie/error.html')
        else:
            return render(request, 'Charlie/loader.html')
