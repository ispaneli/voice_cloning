# Voice cloning

The final work of the course "Python Developer. Professional development (2020)".

## Веб часть проекта
Оболочка представялет собой сайт с базовым функционалом регистрации пользователя.

## Пользователь
***Какие возможности будут у пользователя:***
* Выбрать голос из предоставленного **банка голосов**;
* Написать текст, **озвучку** которого пользователь **хочет получить**;
* Загрузить **запись голоса любого человека** на наш сервис и получить озвучку текста этим голосом.

***Как это выглядит со стороны пользователя:***
1) Пользователь переходит на наш **сайт**;
2) Пользователь **выбирает голос** для озвучки из нашего банка голосов или загружает свою запись голоса;
3) Пользователь к текстовом окне **пишет** "сценарий голосового сообщения", который он хочет получить;
4) Пользователь **получает** запись голоса в формате *.wav* или *.mp3* (скорее всего mp3, т.к. он более легковесный).

***Как запустить:***
1) ставим все из требований
2) не забываем поставить брокер под celery(работало с rabbitmq)
3) в одном терминале запускаем celery
    celery worker -A speech_synthesis --loglevel=info --concurrency 1 -P solo
4) в другом django сервер на необходим ip:port
P.S Если необходимо меняем ip адреса в config
