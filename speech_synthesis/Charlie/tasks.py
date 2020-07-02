from celery import shared_task
import requests
import wave


@shared_task(name="send")
def send_file(address, text, path, username):
    audio = {'audio': open(path, 'rb')}
    export = requests.post(address, data={'text': text}, files=audio)
    f = wave.open('media/recorded_sound' + str(username) + '.wav', 'w')
    f.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
    f.writeframes(export._content)
    f.close()

