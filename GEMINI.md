# Algorytmy w inżynierii danych

## Cel projektu

Celem projektu jest stworzenie własnej biblioteki do automatycznego różniczkowania oraz wykorzystanie jej na przykładzie uczenia konwolucyjnej sieci neuronowej do klasyfikacji obrazów ze zbioru FashionMNIST. Implementacja biblioteki musi być wykonana w języku Julia.

## Etapy projektu

Projekt składa się z dwóch kamieni milowych.

### Kamień milowy 2 (KM1)

Implementacja biblioteki do automatycznego różniczkowania metodą akumulacji w tył (wykorzystując graf obliczeniowy lub generację kodu) pozwalającą na uczenie sieci konwolucyjnej o strukturze przedstawionej w AWID-2026-CNN.ipynb.

Kod biblioteki może być rozwinięciem biblioteki omawianej w trakcie wykładu, dostarczonej przez prowadzącego. Należy samodzielnie rozszerzyć bibliotekę o warstwy Conv, DropOut, MaxPool, oraz o funkcję aktywacji SoftMax.

Sieć powinna po trzech epokach osiągnąć dokładność na zbiorze testowym ok. 85% (± 5 punktów procentowych) w czasie do 15 minut.

Implementacja musi pozwalać na wprowadzanie zmian w parametrach uczenia i architekturze sieci bez konieczności modyfikowania kodu źródłowego modułu oraz łatwe rozszerzenie funkcjonalności bibliotek o kolejne warstwy, funkcje aktywacji lub algorytmy uczące.

### Kamień milowy 2 (KM2)

Implementacja wersji biblioteki pozwalającej rozwiązać problem z KM1, zoptymalizowanej pod kątem złożoności obliczeniowej, gospodarki pamięcią, oraz efektywności wygenerowanego kodu. W przypadku zespołów dwuosobowych, powstaje tylko jedna (wspólna) wersja biblioteki, powstała na podstawie kodu do KM1. Oprócz kodu biblioteki, efektem tego kamienia milowego jest artykuł w języku angielskim (do 4 stron w formatce IEEE), który skupia się wyłącznie na optymalizacjach biblioteki oraz ich weryfikacji. Struktura sieci, parametry algorytmu optymalizacji oraz limity na dokładność na zbiorze testowym nie zmieniają się. W ramach tego kamienia milowego, musi powstać też dodatkowe, referencyjne rozwiązanie w PyTorch, Keras, lub TensorFlow, z którym porównana zostanie zoptymalizowana biblioteka w Julia.

## Instrukcje dla AI

Wykonuj tylko to, o co poproszono. Pisz prosty, zrozumiały kod. Nie rób zbyt wielu zmian jednocześnie.

## FAQ

Q: Czy wymagania co do dokładności są bezwzględne? Czyli sieć bezwzględnie musi osiągnąć minimum 85% accuracy z podanym marginesem w trzech epokach i w mniej niż 15 minut?

A: Najlepiej, gdyby udało się osiągnąć 85%. Ale w uzasadnionych przypadkach uznane zostanie też inny wynik powyżej 80%. Bardziej restrykcyjnie proszę podejść do czasu uczenia i nie przekraczać 15 minut na trzy epoki.

Q: Czy w KM1 akceptowane jest zastosowanie naiwnych implementacji i ich podmiana w KM2? Np. naiwna implementacja warstwy konwolucyjnej.

A: Tak, z ograniczeniem na czas działania. Nawet naiwna implementacja konwolucji powinna pozwolić na 5 min na epokę.

Q: Czy w KM1 wymagana jest obsługa batchy, czy można działać na pojedynczych obrazach? Jeśli nie to czy w KM2 już tak?

A: Na żadnym etapie nie musicie zrobić obsługi batchy na poziomie sieci. Dokładniej mówiąc: nie musicie przetwarzać przez sieć porcji obrazów, tj. tensorów o wymiarach (width x height x channels x batchsize). Wystarczy przetwarzanie pojedynczych obrazów (width x height x channels), natomiast wagi uaktualniamy po zaprezentowaniu sieci batchsize obrazów. Warto przeskalować wartości zakumulowanych pochodnych przez liczbę obrazów w porcji, aby nie trzeba było modyfikować współczynnika uczenia.

Q: Czy potrzebne jest zaimplementowanie specjalnego inicjalizowania wag w warstwie konwolucyjnej?

A: Tak. Inicjalizacja He jest bardzo dobrym pomysłem, natomiast ogranicza się to głównie do przeskalowania wyników z rozkładu normalnego.