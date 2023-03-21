from celery import shared_task
import requests
import wave


@shared_task(name="send")
def send_file(address, text, path, username):
    """
        Асинхронно отправляем запись на сервер. В ответ получим
        файл, который будем хранить локально

    :param address: адрес нашего сервера
    :param text: текст, который надо озвучить
    :param path: путь к сохраненному образцу голоса
    :param username: имя пользователя
    :return: None
    """
    audio = {'audio': open(path, 'rb')}
    export = requests.post(address, data={'text': text}, files=audio)
    if not export.ok:
        raise ValueError(f"{export.status_code}")
    f = wave.open('media/recorded_sound' + str(username) + '.wav', 'w')
    f.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
    f.writeframes(export._content)
    f.close()
    return "OK"

