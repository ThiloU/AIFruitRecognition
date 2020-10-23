from gtts import gTTS

tts = gTTS('Das Bild zeigt einen Apfel', lang='de')
tts.save('./audio/apple.mp3')

tts = gTTS('Das Bild zeigt eine Banane', lang='de')
tts.save('./audio/banana.mp3')

tts = gTTS('Das Bild zeigt eine Orange', lang='de')
tts.save('./audio/orange.mp3')

tts = gTTS('Das Bild zeigt eine Tomate', lang='de')
tts.save('./audio/tomato.mp3')
