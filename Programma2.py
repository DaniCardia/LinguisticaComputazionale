import nltk
import re
import codecs
import sys


listaCaratteri = [".", ",", ";", ":", "!", "?",
                  "(", ")", "[", "]", "-", "*", "/", "'", "<", ">", "#", '‘', '’', '—', '“', '”', '_']

# La funzione TextToList prende in input una lista di frasi e restituisce una lista di parole, una lista di PoS token e una lista di token
def TextToList(frasi):
    paroleTot = []
    tokensPOStot = []
    tokensTot = []
    # Per ogni frase contenuta nel corpus
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        token = []
        # Per ogni token contenuto nella frase tokenizzata
        for token in tokens:
            # Non vengono considerati i caratteri speciali
            if (token not in listaCaratteri):
                # Il token viene aggiunto alla lista totale delle parole
                paroleTot.append(token)
            # tokensTot conterrà tutti i token inclusi i caratteri speciali
            tokensTot.append(token)
        # Vengono generate le part of speech di ogni tokens
        tokensPOS = nltk.pos_tag(tokens)
        # Vengono sommati i Pos tokens della frase attuale con quella precedente
        tokensPOStot += tokensPOS
    # Vengono restituiti i tokens totali e le POS totali
    return paroleTot, tokensPOStot, tokensTot

# La funzione personeInFrase prende in input le posTokens e restituisce una lista di nomi di persona
def personeInFrase(posTokens):
    nomiPersona = []
    # Viene eseguita la classificazione delle entità nominate
    albero = nltk.ne_chunk(posTokens)
    # Viene trasformato l'albero in formato IOB
    IOBFormat = nltk.chunk.tree2conllstr(albero)
    # Per ogni nodo contenuto in albero
    for nodo in albero:
        NE = ''
        # Se chunck è un nodo intermedio
        if hasattr(nodo, 'label'):
            # Se il nodo è relativo ad un nome di persona
            if nodo.label() in ['PERSON']:
                # Gestione dei nomi composti da più parole
                for i, partNE in enumerate(nodo.leaves()):
                    # Se è un nome composto
                    if len(nodo.leaves()) - i > 1:
                        NE += partNE[0]+' '
                    # Se è rimasto un solo termine
                    else:
                        NE = NE+partNE[0]
                # Viene aggiunto il nome alla lista dei nomi di persona
                nomiPersona.append(NE)
        # Viene restituita la lista dei nomi di persona
    return nomiPersona

# La funzione NomiPersonaFrequenti prende in input le posTokens e restituisce i nomi più frequenti
def NomiPersonaFrequenti(posTokens):
    # Viene ottenuta la lista delle persona nella frase con token pos
    nomiPersona = personeInFrase(posTokens)
    # Viene calcolata la frequenza dei nomi di persona
    freqNomi = nltk.FreqDist(nomiPersona)
    # Vengono restituiti i 10 nomi di più frequenti
    return freqNomi.most_common(10)

# La funzione FrasiConNomi prende in input le frasi e un nomeProprio e restituisce le frasi che contengono il nome proprio
def FrasiConNomi(frasi, nomeProprio):
    fraseConNome = []
    # Per ogni frase in frasi
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        # Vengono generate le part of speech di ogni tokens
        tokensPOS = nltk.pos_tag(tokens)
        # Viene chiamata la funzione che restituisce la lista delle persone contenute nella frase tokenizzata
        listaPersone = personeInFrase(tokensPOS)
        # Se nomeProprio è contenuto nella frase tokenizzata, viene aggiunta tale frase alla lista delle frasi da restituire
        if nomeProprio in listaPersone:
            # La frase viene aggiunta alla lista delle frasi da restituire
            fraseConNome.append(frase)
    # Vengono restituite le frasi contenenti il nome proprio
    return fraseConNome

# La funzione FrasePiuLunga prende in input le frasi e restituisce la frase più lunga
def FrasePiuLunga(frasi):
    maxLunghezzaFrase = -1
    frasePiuLunga = ''
    # Per ogni frase contenuta in frasi
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        # Viene calcolata la lunghezza della frase in base ai tokens
        lunghezzaFrase = len(tokens)
        # Se lunghezzaFrase è maggiore di maxLunghezzaFrase
        if lunghezzaFrase > maxLunghezzaFrase:
            # Aggiorna la frase più lunga
            maxLunghezzaFrase = lunghezzaFrase
            frasePiuLunga = frase
    # Viene restituita la frase più lunga
    return frasePiuLunga

# La funzione FrasePiuBreve prende in input le frasi e restituisce la frase più lunga
def FrasePiuBreve(frasi):
    minLunghezzaFrase = float('inf')
    frasePiuBreve = ''
    # Per ogni frase contenuta in frasi
    for frase in frasi:
        # La frase viene divisa in token
        tokens = nltk.word_tokenize(frase)
        # Viene calcolata la lunghezza della frase in base ai tokens
        lunghezzaFrase = len(tokens)
        # Se lunghezzaFrase è minore di minLunghezzaFrase
        if lunghezzaFrase < minLunghezzaFrase:
            # Aggiorna la frase più breve
            minLunghezzaFrase = lunghezzaFrase
            frasePiuBreve = frase
    # Viene restituita la frase più corta
    return frasePiuBreve

# La funzione LuoghiPiuFrequenti prende in input le posTokens e restituisce i luoghi più frequenti
def LuoghiPiuFrequenti(posTokens):
    luoghi = []
    # Viene eseguita la classificazione delle entità nominate
    albero = nltk.ne_chunk(posTokens)
    # Viene trasformato l'albero in formato IOB
    IOBFormat = nltk.chunk.tree2conllstr(albero)
    # Per ogni nodo contenuto in albero
    for nodo in albero:
        NE = ''
        # Se chunck è un nodo intermedio
        if hasattr(nodo, 'label'):
            # Verifico che il nodo contenga informazioni geografiche
            if nodo.label() in ['GPE']:
                # Gestione posizione geografiche composte da più parole
                for i, partNE in enumerate(nodo.leaves()):
                    if len(nodo.leaves()) - i > 1:
                        NE += partNE[0]+' '
                    else:
                        NE = NE+partNE[0]
                        # Viene aggiunto il nodo alla lista dei luoghi
                luoghi.append(NE)
    freqLuoghi = nltk.FreqDist(luoghi)
    # Vengono restituiti i 10 luoghi più frequenti
    return freqLuoghi.most_common(10)

# La funzione PersonePiuFrequenti prende in input le posTokens e restituisce le persone più frequenti
def PersonePiuFrequenti(posTokens):
    persone = []
    # Viene eseguita la classificazione delle entità nominate
    albero = nltk.ne_chunk(posTokens)
    # Viene trasformato l'albero in formato IOB
    IOBFormat = nltk.chunk.tree2conllstr(albero)
    # Per ogni nodo contenuto in albero
    for nodo in albero:
        NE = ''
        # Se chunck è un nodo intermedio
        if hasattr(nodo, 'label'):
            # Se il nodo è relativo ad un nome di persona
            if nodo.label() in ['PERSON']:
                # Gestione dei nomi composti da più parole
                for i, partNE in enumerate(nodo.leaves()):
                    # Se è un nome composto
                    if len(nodo.leaves()) - i > 1:
                        NE += partNE[0]+' '
                    else:
                        NE = NE+partNE[0]
                        # Viene aggiunto il nome alla lista dei nomi di persona
                persone.append(NE)
        # Viene calcolata la frequeza dei nomi di persona
    freqPersone = nltk.FreqDist(persone)
    # Vengono restituite le 10 persone più frequenti
    return freqPersone.most_common(10)

# La funzione SostantiviPiuFrequenti prende in input le posTokens e restituisce i sostantivi più frequenti
def SostantiviPiuFrequenti(posTokens):
    sostantivi = []
    # Per ogni tokens presente in posTokens
    for tokens in posTokens:
        # Viene verificato che la POS, presa in considerazione, sia un sostantivo
        if tokens[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            # Non vengono considerati i caratteri speciali
            if tokens[0] not in listaCaratteri:
                # Vengono aggiunti i sostantivi senza le POS
                sostantivi.append(tokens[0])
    # Viene calcolata la frequenza dei sostantivi
    freqSostantivi = nltk.FreqDist(sostantivi)
    # Vengono restituiti i 10 sostantivi più frequenti
    return freqSostantivi.most_common(10)

# La funzione VerbiPiuFrequenti prende in input le posTokens e restituisce i verbi più frequenti
def VerbiPiuFrequenti(posTokens):
    verbi = []
    # Per ogni tokens presente in posTokens
    for tokens in posTokens:
        # Viene verificato che la POS, presa in considerazione, sia un verbo
        if tokens[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            # Non vengono considerati i caratteri speciali
            if tokens[0] not in listaCaratteri:
                # Vengono aggiunti i verbi senza le POS
                verbi.append(tokens[0])
    # Viene calcolata la frequenza dei verbi
    freqVerbi = nltk.FreqDist(verbi)
    # Vengono restituiti i 10 verbi più frequenti
    return freqVerbi.most_common(10)

# La funzione EstrazioneDate prende in input le frasi e restituisce una lista conetenente le date, i mesi e i giorni della settimana che compaiono nelle frasi
def EstrazioneDate(frasi):
    # Vengono definite delle espressioni regolari per il ritrovamento delle date
    patternData = r'(\d+-\d+-\d+) | (\d+/\d+/\d+)'
    # Vengono definite delle espressioni regolari per il ritrovamento dei mesi
    patternMese = r'(?:Jan(?:uary)|Feb(?:ruary)|Mar(?:ch)|Apr(?:il)|May|Jun(?:e)|Jul(?:y)|Aug(?:ust)|Sept(?:ember)|Oct(?:ober)|Nov(?:ember)|Dec(?:ember))'
    # Vengono definite delle espressioni regolari per il ritrovamento dei giorni della settimana
    patternGiorno = r'(?:Mon(?:day)|Tue(?:sday)|Wed(?:nesday)|Thu(?:rsday)|Fri(?:day)|Sat(?:urday)|Sun(?:day))'
    date = []
    # Per ogni frase in frasi
    for frase in frasi:
        # cerca nella frase le espressioni regolari
        date += re.findall(patternData, frase)
        date += re.findall(patternMese, frase)
        date += re.findall(patternGiorno, frase)
    return date

# La funzione ProbabilitaModelloMarkov prende in input la lunghezzaCorpus, la distribuzioneFrequenza e la frase e restituisce la probabilità secondo il modello di Markov di grado 0
def ProbabilitaModelloMarkov(lunghezzaCorpus, distribuzioneFrequenza, frase):
    probabilita = 1.0
    # Per ogni token presente in frase
    for token in frase:
        # Calcola la probabilità
        probabilitaToken = distribuzioneFrequenza[token] * \
            1.0/(lunghezzaCorpus*1.0)
        probabilita = probabilita*probabilitaToken
    # Restituisce la probabilità secondo il modello di Markov di grado 0
    return probabilita

# La funzione EstrazioneFrasiAltaProb prende in input le frasi e il corpus composto da tokens e restituisce la frase con più alta probabilità
def EstrazioneFrasiAltaProb(frasi, tokensCorpus):
    lunghezzaFrase = 0.0
    probabilita = 'Nessuna frase con lunghezza adatta'
    freqCorpus = nltk.FreqDist(tokensCorpus)
    # Viene creato un dizionario
    dizionarioFrequenze = {}
    # Per ogni frase in frasi
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        # Viene calcolata la lunghezza della frase
        lunghezzaFrase = len(tokens)
        probabilita = 'Nessuna frase con lunghezza adatta'
        # Se la lunghezza della frase è maggiore di 7 e minore di 13
        if (lunghezzaFrase > 7) and (lunghezzaFrase < 13):
            # Viene calcolata la probabilità secondo il modello di markov di grado 0 utilizzando la relativa funzione
            probabilita = ProbabilitaModelloMarkov(
                len(tokensCorpus), freqCorpus, tokens)
            dizionarioFrequenze[frase] = probabilita
    # Viene ordinata la lista
    listaOrdinata = sorted(dizionarioFrequenze.items(),
                           key=lambda coppia: coppia[1], reverse=True)
    # Se non ci sono frasi di lunghezza compresa tra 7 e 13
    if listaOrdinata == []:
        return []
    # Restituisce la frase con più alta probabilità
    return listaOrdinata[0]


def main(files):
    # Verifica che siano presenti i parametri
    if len(files) < 2:
        print("Inserire i file come parametri")
        return
    # Rimozione del nome del programma dalla lista dei file
    files.pop(0)
    for file in files:
        # Lettura del file
        fileInput = codecs.open(file, "r", "utf-8")
        raw = fileInput.read()
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        frasi = sent_tokenizer.tokenize(raw)
        # Trasformarzione delle frasi in tokens e tokens con pos
        listaParole, list_posTokens, listaTokens = TextToList(frasi)
        # Estrazione nomi propri più frequenti
        nomiPiuFrequenti = NomiPersonaFrequenti(list_posTokens)
        # Stampo i 10 nomi propri più frequenti
        print('I 10 nomi più frequenti del',
              (file), 'sono', (nomiPiuFrequenti))
        for nomeProprio in nomiPiuFrequenti:
            nomeProprio = nomeProprio[0]
            frasiConNomeProprio = FrasiConNomi(frasi, nomeProprio)
            fraseLunga = FrasePiuLunga(frasiConNomeProprio)
            fraseCorta = FrasePiuBreve(frasiConNomeProprio)
            frasiSelezionate, frasiSelezionatePOS, _ = TextToList(
                frasiConNomeProprio)
            luoghi = LuoghiPiuFrequenti(frasiSelezionatePOS)
            persone = PersonePiuFrequenti(frasiSelezionatePOS)
            sostantivi = SostantiviPiuFrequenti(frasiSelezionatePOS)
            verbi = VerbiPiuFrequenti(frasiSelezionatePOS)
            dateMesiGiorni = EstrazioneDate(frasiConNomeProprio)
            frasiAltaProb = EstrazioneFrasiAltaProb(
                frasiConNomeProprio, listaTokens)
            # Stampo la frase più lunga
            print('La frase più lunga del', (file),
                  ' che contiene il nome ', nomeProprio, 'è:', (fraseLunga))
            # Stampo la frase più breve
            print('La frase più breve del', (file),
                  ' che contiene il nome ', nomeProprio, 'è:', (fraseCorta))
            # Stampo i 10 luoghi più frequenti
            print('I 10 luoghi più frequenti del ', (file),
                  ' e nelle frasi che contenengono il nome ', nomeProprio, 'sono:', (luoghi))
            # Stampo le 10 persone più frequenti
            print('Le 10 persone più frequenti del', (file),
                  ' e nelle frasi che contenengono il nome ', nomeProprio, 'sono:', (persone))
            # Stampo i 10 sostantivi più frequenti
            print('I 10 sostantivi più frequenti del', (file),
                  ' e nelle frasi che contenengono il nome ', nomeProprio, 'sono:', (sostantivi))
            # Stampo i 10 verbi più frequenti
            print('I 10 verbi più frequenti del', (file),
                  ' e nelle frasi che contenengono il nome ', nomeProprio, 'sono:', (verbi))
            # Stampo le date, i mesi e i giorni della settimana
            print('Le date, i mesi e i giorni della settimana del ', (file),
                  ' e nelle frasi che contenengono il nome ', nomeProprio, 'sono', (dateMesiGiorni))
            # Stampo la frase con probabilità più attraverso il modello di Markov di ordine 0
            print('La frase con probabilità più alta del ', (file),
                  ' e nelle frasi che contengono il nome', nomeProprio, 'sono', (frasiAltaProb))


main(sys.argv)
