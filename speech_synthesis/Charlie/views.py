from django.shortcuts import render, redirect, reverse
from Charlie.models import Category, Page, UserProfile, Synthesizer
from django.contrib.auth.models import User
from Charlie.forms import CategoryForm, PageForm, UserForm, UserProfileForm, SynthesizerForm
from django.contrib.auth.decorators import login_required
from datetime import datetime
from django.views import View
from django.utils.decorators import method_decorator


# Create your views here.


def get_server_side_cookie(request, cookie, default_val=None):
    val = request.session.get(cookie)
    if not val:
        val = default_val
    return val


def visitor_cookie_handler(request):
    visits = int(get_server_side_cookie(request, 'visits', '1'))
    last_visit_cookie = get_server_side_cookie(request, 'last_visit',
                                               str(datetime.now())
                                               )
    last_visit_time = datetime.strptime(last_visit_cookie[:-7], '%Y-%m-%d %H:%M:%S')

    if (datetime.now() - last_visit_time).seconds > 0:
        visits += 1
        request.session['last_visit'] = str(datetime.now())
    else:
        request.session['last_visit'] = last_visit_cookie

    request.session['visits'] = visits


class Index(View):
    @staticmethod
    def get(request):
        category_list = Category.objects.order_by('-likes')[:5]
        page_list = Page.objects.order_by('-views')[:5]
        context_dict = {'boldmessage': 'Crunchy, creamy, cookie, candy, cupcake!',
                        'categories': category_list,
                        'pages': page_list}
        visitor_cookie_handler(request)
        response = render(request, 'Charlie/index.html', context=context_dict)
        return response


class AboutView(View):
    @staticmethod
    def get(request):
        context_dict = {}
        visitor_cookie_handler(request)
        context_dict['visits'] = request.session['visits']

        return render(request, 'Charlie/about.html', context=context_dict)


class ShowCategory(View):
    @staticmethod
    def get(request, category_name_slug):
        context_dict = {}
        try:
            category = Category.objects.get(slug=category_name_slug)
            pages = Page.objects.filter(category=category)
            context_dict['pages'] = pages
            context_dict['category'] = category
        except Category.DoesNotExist:
            context_dict['pages'] = None
            context_dict['category'] = None

        return render(request, 'Charlie/category.html', context=context_dict)


class AddCategoryView(View):
    @method_decorator(login_required)
    def get(self, request):
        form = CategoryForm()
        return render(request, 'Charlie/add_category.html',
                      {'form': form})

    @method_decorator(login_required)
    def post(self, request):
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save(commit=True)
            return redirect('/Charlie/')
        else:
            print(form.errors)

        return render(request, 'Charlie/add_category.html',
                      {'form': form})


def add_page_decorator(func):
    def wrapper(request, category_name_slug):
        try:
            category = Category.objects.get(slug=category_name_slug)
        except Category.DoesNotExist:
            category = None
        if category is None:
            return redirect('/Charlie/')
        form = PageForm()
        context_dict = {'form': form, 'category': category}
        return func(request, category_name_slug, context_dict, category)

    return wrapper


class AddPageView(View):

    @method_decorator(login_required)
    @method_decorator(add_page_decorator)
    def get(self, request, category_name_slug, context_dict, category):
        return render(request, 'Charlie/add_page.html', context=context_dict)

    @method_decorator(login_required)
    @method_decorator(add_page_decorator)
    def post(self, request, category_name_slug, context_dict, category):
        form = PageForm(request.POST)
        if form.is_valid():
            page = form.save(commit=False)
            page.category = category
            page.views = 0
            page.save()

            return redirect(reverse('Charlie:show_category',
                                    kwargs={'category_name_slug': category_name_slug}))
        else:
            print(form.errors)
        return render(request, 'Charlie/add_page.html', context=context_dict)


class RegisterProfileView(View):
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


class ProfileView(View):

    @staticmethod
    def get_user_details(username):
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return None

        user_profile = UserProfile.objects.get_or_create(user=user)[0]
        form = UserProfileForm({'website': user_profile.website,
                                'picture': user_profile.picture})

        return user, user_profile, form

    @method_decorator(login_required)
    def get(self, request, username):
        try:
            user, user_profile, form = self.get_user_details(username)
            synthesizer = Synthesizer.objects.get_or_create(user=user)[0]
        except TypeError:
            return redirect(reverse('Charlie:index'))

        context_dict = {'user_profile': user_profile,
                        'selected_user': user,
                        'form': form,
                        'synthesizer': synthesizer}

        return render(request, 'Charlie/profile.html', context_dict)

    @method_decorator(login_required)
    def post(self, request, username):
        try:
            user, user_profile, form = self.get_user_details(username)
        except TypeError:
            return redirect(reverse('Charlie:index'))

        form = UserProfileForm(request.POST, request.FILES, instance=user_profile)
        if form.is_valid():
            form.save(commit=True)
            return redirect('Charlie:profile', user.username)
        else:
            print(form.errors)
        context_dict = {'user_profile': user_profile,
                        'selected_user': user,
                        'form': form}

        return render(request, 'Charlie/profile.html', context_dict)


class ListProfilesView(View):
    @method_decorator(login_required)
    def get(self, request):
        profiles = UserProfile.objects.all()

        return render(request, 'Charlie/list_profiles.html',
                      context={'user_profile_list': profiles})


class SynthesizerView(View):
    path = ''
    @staticmethod
    def get_user_details(username):
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return None

        synthesizer = Synthesizer.objects.get_or_create(user=user)[0]
        form = SynthesizerForm({'text': synthesizer.text,
                                'input_audio': synthesizer.input_audio})

        return user, synthesizer, form

    @method_decorator(login_required)
    def get(self, request):
        try:
            user, synthesizer, form = self.get_user_details(request.user)
        except TypeError:
            return redirect(reverse('Charlie:index'))

        context_dict = {'synthesizer': synthesizer,
                        'form': form,
                        }
        return render(request, 'Charlie/synthesizer.html', context=context_dict)

    @method_decorator(login_required)
    def post(self, request):
        try:
            user, synthesizer, form = self.get_user_details(request.user)
        except TypeError:
            return redirect(reverse('Charlie:index'))

        form = SynthesizerForm(request.POST, request.FILES, instance=synthesizer)
        if form.is_valid():
            form.save(commit=True)
            return redirect('Charlie:synthesizer')
        else:
            print(form.errors)

        context_dict = {'synthesizer': synthesizer,
                        'form': form}
        return render(request, 'Charlie/synthesizer.html', context=context_dict)