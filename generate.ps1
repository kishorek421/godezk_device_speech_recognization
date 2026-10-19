Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("algorithm_test.wav")
$s.Speak("algorithm")
$s.Dispose()
