#' ---
#' title: "System Klasyfikacji: Predykcja Decezji Fed (SVM + 5-Fold CV)"
#' author: "Anatoliy Daneiko, Mikołaj Krawiec, Stefan Łaszkiewicz"
#' output:
#'   html_document:
#'     theme: readable      
#'     highlight: kate      
#'     toc: true            
#'     toc_float: true
#' ---

knitr::opts_chunk$set(message = FALSE, warning = FALSE)

#' # Wymagane pakiety
library(tm)
library(tidyverse)
library(tidytext)
library(ggplot2)
library(ggthemes)
library(SnowballC)
library(caret)   
library(e1071)   
library(wordcloud)
library(RColorBrewer)
library(textdata)

#' # 1. Wgrywanie danych
# Wczytanie pliku 3-kolumnowego
dane <- read.csv("przemowienia_fed.csv", stringsAsFactors = FALSE, encoding = "UTF-8")

# Budowa korpusu bezpośrednio z kolumny z tekstem przemówienia
korpus <- VCorpus(VectorSource(dane$tekst_przemowienia))

#' # 2. Przetwarzanie i oczyszczanie tekstu
korpus <- tm_map(korpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))
doSpacji <- content_transformer(function (x, pattern) gsub(pattern, " ", x))

rozklej_slowa <- content_transformer(function(x) {
  x <- gsub("([a-z])([A-Z])", "\\1 \\2", x) # Rozkleja małą i wielką literę stykające się bezpośrednio
  x <- gsub("([\\.\\?!,])([^[:space:]0-9])", "\\1 \\2", x) # Wymusza spację po znakach interpunkcyjnych
  return(x)
})
korpus <- tm_map(korpus, rozklej_slowa)

korpus <- tm_map(korpus, doSpacji, "-")
korpus <- tm_map(korpus, doSpacji, "—")
korpus <- tm_map(korpus, doSpacji, "/")

# Usuwanie adresów URL i wzmianek
korpus <- tm_map(korpus, doSpacji, "(s?)(f|ht)tp(s?)://\\S+\\b")
korpus <- tm_map(korpus, doSpacji, "http\\w*")
korpus <- tm_map(korpus, doSpacji, "www")
korpus <- tm_map(korpus, doSpacji, "@\\w+")

# Usuwanie niestandardowych cudzysłowów
korpus <- tm_map(korpus, doSpacji, "[„”„“'’‘\"区域“”]")

# Konwersja wielkości liter
korpus <- tm_map(korpus, content_transformer(tolower))


# Standardowe usuwanie znaków
korpus <- tm_map(korpus, removeNumbers)
korpus <- tm_map(korpus, removePunctuation)

# Usuwanie stop words i slangu proceduralnego
korpus <- tm_map(korpus, removeWords, stopwords("english"))
korpus <- tm_map(korpus, removeWords, c("chairman", "committee", "ranking", "member", "testimony", "federal", "reserve", "board"))

# Czyszczenie wielokrotnych białych znaków na samym końcu
korpus <- tm_map(korpus, stripWhitespace)

#' # 3. Stemming i Uzupełnianie Rdzeni
korpus_kopia <- korpus

slownik_slow <- unique(unlist(strsplit(sapply(korpus_kopia, as.character), "[[:space:]\n\r]+")))
slownik_slow <- slownik_slow[slownik_slow != ""] # Usunięcie pustych elementów

# Wykonanie bazowego stemowania
korpus_rdzenie <- tm_map(korpus, stemDocument)

# Definicja funkcji uzupełniającej rdzenie
uzupelnij_rdzenie <- content_transformer(function(x, slownik) {
  x <- unlist(strsplit(x, " "))                  
  x <- stemCompletion(x, dictionary = slownik, type = "longest") 
  paste(x, collapse = " ")                        
})

korpus_uzupelniony <- tm_map(korpus_rdzenie, uzupelnij_rdzenie, slownik = slownik_slow)
korpus_uzupelniony <- tm_map(korpus_uzupelniony, doSpacji, "NA")
korpus_uzupelniony <- tm_map(korpus_uzupelniony, stripWhitespace)

#' # 4. Analiza Sentymentu (Słownik Finansowy Loughran-McDonald)
tdm_dla_sentymentu <- TermDocumentMatrix(korpus_uzupelniony)

# Agregacja unikalnych identyfikatorów dla wszystkich dokumentów w korpusie
wszystkie_dokumenty <- data.frame(document = as.character(1:length(korpus_uzupelniony)))

sentyment_dokumentow <- tidy(tdm_dla_sentymentu) %>%
  rename(word = term) %>%
  inner_join(get_sentiments("loughran"), by = "word", relationship = "many-to-many") %>%
  filter(sentiment %in% c("positive", "negative")) %>%
  group_by(document, sentiment) %>%
  # Agregacja łącznej liczby wystąpień słów emocjonalnych w dokumentach
  summarise(n = sum(count), .groups = "drop") %>% 
  spread(sentiment, n, fill = 0) %>%
  mutate(SENTYMENT_NETTO = (positive - negative) / (positive + negative + 1))

# Mapowanie wyników z uwzględnieniem dokumentów o neutralnym wydźwięku
pelny_sentyment <- wszystkie_dokumenty %>%
  left_join(sentyment_dokumentow, by = "document") %>%
  mutate(
    positive = ifelse(is.na(positive), 0, positive),
    negative = ifelse(is.na(negative), 0, negative),
    SENTYMENT_NETTO = ifelse(is.na(SENTYMENT_NETTO), 0, SENTYMENT_NETTO)
  )

wektor_sentymentu <- pelny_sentyment$SENTYMENT_NETTO

#' # 5. Połączenie TF-IDF i Sentymentu
macierz_tfidf <- TermDocumentMatrix(korpus_uzupelniony,
                                    control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

# Redukcja wymiarowości poprzez usunięcie rzadkich tokenów (poniżej 15% dokumentów)
macierz_tfidf <- removeSparseTerms(macierz_tfidf, 0.85)

# Transpozycja i konwersja macierzy rzadkiej na ramkę danych obiektów
ramka_cech <- as.data.frame(t(as.matrix(macierz_tfidf)))

# Standaryzacja nazw kolumn pod kątem wymagań walidacji pakietu caret
names(ramka_cech) <- make.names(names(ramka_cech))
names(ramka_cech) <- make.unique(names(ramka_cech))

# Integracja wektora sentymentu Loughran-McDonald jako zmiennej objaśniającej
ramka_cech$CECHA_SENTYMENT_LOUGHRAN <- wektor_sentymentu

# Definicja zmiennej zależnej (target) jako faktora z ustalonymi poziomami
ramka_cech$Zmienna_Celowa <- factor(dane$kierunek_polityki, levels = c("luzowanie", "zaciesnianie"))

#' # 6. Uczenie Maszynowe: 5-Fold Cross-Validation (SVM)
# Konfiguracja procesu 5-krotnej walidacji krzyżowej
kontrola_treningu <- trainControl(method = "cv", 
                                  number = 5,
                                  savePredictions = "final")

# Estymacja liniowego klasyfikatora SVM wraz z centrowaniem i skalowaniem cech
set.seed(123) 
model_svm_cv <- train(Zmienna_Celowa ~ ., 
                      data = ramka_cech, 
                      method = "svmLinear", 
                      preProcess = c("center", "scale"), 
                      trControl = kontrola_treningu)

#' # 7. Oceny, Macierz Pomyłek i Wizualizacje
wyniki_predykcji <- model_svm_cv$pred

# Standaryzacja poziomów czynnika dla wartości prognozowanych i obserwowanych
poziomy <- c("luzowanie", "zaciesnianie")
wyniki_predykcji$pred <- factor(wyniki_predykcji$pred, levels = poziomy)
wyniki_predykcji$obs  <- factor(wyniki_predykcji$obs, levels = poziomy)

# Generowanie kwadratowej macierzy pomyłek o stałym wymiarze 2x2
macierz_pomylek <- table(Przewidywane = wyniki_predykcji$pred, 
                         Rzeczywiste = wyniki_predykcji$obs)

print("Połączona Macierz Pomyłek dla 5-Fold CV:")
print(macierz_pomylek)

# Wyznaczenie podstawowych komponentów macierzy błędów
TP <- macierz_pomylek["zaciesnianie", "zaciesnianie"]
TN <- macierz_pomylek["luzowanie", "luzowanie"]
FP <- macierz_pomylek["zaciesnianie", "luzowanie"]
FN <- macierz_pomylek["luzowanie", "zaciesnianie"]

# Obliczenie rzetelnych metryk klasyfikacji z obsługą dzielenia przez zero
precyzja   <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
czulosc    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
specyfika  <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
dokladnosc <- (TP + TN) / sum(macierz_pomylek)

wynik_f1   <- ifelse((precyzja + czulosc) == 0, 0, 
                     2 * (precyzja * czulosc) / (precyzja + czulosc))

# Przygotowanie struktury danych do wizualizacji metryk
ramka_metryk <- data.frame(
  Metryka = c("Dokładność", "Precyzja", "Czułość", "Specyfika", "Wynik F1"),
  Wartosc = c(dokladnosc, precyzja, czulosc, specyfika, wynik_f1)
)

ggplot(ramka_metryk, aes(x = Metryka, y = Wartosc, fill = Metryka)) +
  geom_col(width = 0.5, color = "black") +
  geom_text(aes(label = round(Wartosc, 2)), vjust = -0.5, size = 5) +
  ylim(0, 1) +
  labs(title = "Wydajność Klasyfikatora SVM (Walidacja Krzyżowa)", y = "Wartość", x = "") +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Przekształcenie tabeli do formatu długiego pod wykres
ramka_pomylek <- as.data.frame(as.table(macierz_pomylek))

ramka_pomylek$Przewidywane <- as.character(ramka_pomylek$Przewidywane)
ramka_pomylek$Rzeczywiste <- as.character(ramka_pomylek$Rzeczywiste)

# Mapowanie logiczne stanów macierzy i generowanie etykiet tekstowych
ramka_pomylek <- ramka_pomylek %>%
  mutate(Typ_Wyniku = case_when(
    Przewidywane == Rzeczywiste & Przewidywane == "zaciesnianie" ~ "TRAFIONE (TP)",
    Przewidywane == Rzeczywiste & Przewidywane == "luzowanie"     ~ "TRAFIONE (TN)",
    Przewidywane != Rzeczywiste & Przewidywane == "zaciesnianie" ~ "BŁĄD (FP)",
    Przewidywane != Rzeczywiste & Przewidywane == "luzowanie"     ~ "BŁĄD (FN)"
  )) %>%
  mutate(Kompaktowa_Etykieta = paste0(Typ_Wyniku, "\n", Freq, " raport(y)")) %>%
  mutate(Status = ifelse(Przewidywane == Rzeczywiste, "OK", Typ_Wyniku))

ggplot(ramka_pomylek, aes(x = Rzeczywiste, y = Przewidywane, fill = Status)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = Kompaktowa_Etykieta), size = 3.5, fontface = "bold", color = "white") +
  scale_fill_manual(values = c(
    "OK" = "forestgreen",
    "BŁĄD (FP)" = "orange",
    "BŁĄD (FN)" = "red"
  ), guide = "none") + 
  labs(title = "Macierz Pomyłek dla Raportów Fed",
       subtitle = "Zgodność predykcji modelu SVM z rzeczywistością") +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 11, fontface = "bold"),
    axis.text.y = element_text(size = 11, fontface = "bold")
  )

#' ## Wizualizacja 3: Najczęstsze Słowa Sentymentu (Loughran-McDonald)
# Ekstrakcja i filtrowanie top 10 słów dla każdej kategorii emocjonalnej
top_slowa_sentymentu <- tidy(tdm_dla_sentymentu) %>%
  rename(word = term) %>%
  inner_join(get_sentiments("loughran"), by = "word", relationship = "many-to-many") %>%
  filter(sentiment %in% c("positive", "negative")) %>%
  group_by(word, sentiment) %>%
  summarise(n = sum(count), .groups = "drop") %>%
  group_by(sentiment) %>%
  slice_max(n, n = 10, with_ties = FALSE) %>% # Wybór top 10 słów
  ungroup() %>%
  mutate(word = reorder_within(word, n, sentiment)) # Poprawne sortowanie wewnątrz facetów

# Generowanie uporządkowanego wykresu kolumnowego poziomego
ggplot(top_slowa_sentymentu, aes(x = n, y = word, fill = sentiment)) +
  geom_col(color = "black", show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y", 
             labeller = as_labeller(c(positive = "SŁOWA POZYTYWNE", negative = "SŁOWA NEGATYWNE"))) +
  scale_y_reordered() + # Wyświetlenie unikalnych uporządkowanych etykiet osi Y
  scale_fill_manual(values = c("positive" = "forestgreen", "negative" = "darkred")) +
  labs(title = "Kluczowe Słowa Sentymentu Finansowego w Raportach Fed",
       subtitle = "Top 10 najczęstszych tokenów według słownika Loughran-McDonald",
       x = "Łączna liczba wystąpień we wszystkich dokumentach", y = "") +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(size = 11, fontface = "bold", color = "black"),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 10, fontface = "bold"),
    plot.title = element_text(face = "bold", size = 14)
  )

#' ## Wizualizacja 4: Globalna Chmura Słów (na podstawie wag TF-IDF)
# Kalkulacja sumy ważonej oraz inicjalizacja zbioru pod chmurę
wagi_slow <- sort(rowSums(as.matrix(macierz_tfidf)), decreasing = TRUE)
ramka_chmury <- data.frame(slowo = names(wagi_slow), waga = wagi_slow)

# Generowanie chmury słów ze skalowaniem rozmiaru zapobiegającym ucinaniu tekstu
wordcloud(words = ramka_chmury$slowo, 
          freq = ramka_chmury$waga, 
          max.words = 50, 
          random.order = FALSE,
          scale = c(2.5, 0.4), 
          rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))