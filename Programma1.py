import nltk
import sys
import codecs
import math
from itertools import chain

listaCaratteri = [".",",",";",":","!","?","(",")","[","]","-","*","/","'","<",">","#",'‘','’','—','“','”','_']

# La funzione TextToList prende in input una lista di frasi e restituisce una lista di token e una lista di PoS token
def TextToList(frasi):
    tokensTot = []
    tokensPOStot = []
    # Per ogni frase contenuta in frasi
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        token = []
        # Per ogni token contenuto nella frase tokenizzata
        for token in tokens:
            # Non vengono considerati i caratteri speciali
            if (token not in listaCaratteri):
                # Il token viene aggiunto alla lista totale dei token
                tokensTot.append(token)
        # Vengono generate le part of speech di ogni tokens
        tokensPOS = nltk.pos_tag(tokens)
        # Vengono sommati i tokens della frase attuale con quella precedente
        tokensPOStot += tokensPOS
    # Vengono restituiti i tokens totali e le POS totali
    return tokensTot, tokensPOStot

# La funzione CalcolaLunghezza prende in input una lista di frasi e restituisce il numero dei tokens del corpus
def CalcolaLunghezza(frasi):
    lunghezzaTOT = 0.0 
    # Per ogni frase nel testo
    for frase in frasi:
        # La frase viene divisa la frase in tokens
        tokens = nltk.word_tokenize(frase)
        # Viene calcolata la lunghezza totale dei tokens
        lunghezzaTOT += len(tokens)
    # Viene restituita la lunghezza totale del corpus in termini di tokens
    return lunghezzaTOT

# La funzione CalcolaLunghezzaParole prende in input una lista di parole e ne calcola la lunghezza totale in termini di caratteri
def CalcolaLunghezzaParole(parole):
    lunghezzaCaratteri = 0.0
    lunghezzaCaratteriTOT = 0.0
    # Per ogni parola in una frase
    for parola in parole:
        # Viene calcolata la lunghezza dei caratteri di una parola
        lunghezzaCaratteri = len(parola)
        # Vengono sommate le lunghezze dei caratteri delle parole
        lunghezzaCaratteriTOT += lunghezzaCaratteri
    # Viene restituita la lunghezza totale delle parole 
    return lunghezzaCaratteriTOT

# La funzione MediaCaratteri prende in input una lista di frasi e restituisce la media delle parole in caratteri
def MediaCaratteri(frasi):
    lunghezzaCaratteriFrase = 0
    numeroParole = 0
    # Per ogni frase contenuta in frasi 
    for frase in frasi:
        # La frase viene divisa in tokens
        tokens = nltk.word_tokenize(frase)
        # Non vengono considerati i caratteri speciali
        if (tokens not in listaCaratteri):
            # Vengono sommati i numeri dei tokens in una frase
            numeroParole += len(tokens)
            # Viene sommata la lunghezza dei caratteri di una frase
            lunghezzaCaratteriFrase += CalcolaLunghezzaParole(tokens)
    # Viene restituita la media dei caratteri in un testo
    return float(lunghezzaCaratteriFrase/numeroParole)

# La funzione Grandezza Vocabolario prende in input una lista di frasi e restituisce la grandezza del vocabolario
def GrandezzaVocabolario(frasi):
    lunghezzaVocabolario = 0
    # Viene creato un insieme per evitare duplicati
    vocabolario = set()
    # Per ogni frase contenuta in frasi
    for frase in frasi:
        # La frase viene divisa in tokens 
        tokens = nltk.word_tokenize(frase)
        # Vengono rimossi dall'insieme dei tokens i caratteri speciali, attraverso la differenza insiemistica
        token = set(tokens)-set(listaCaratteri)
        # Vengono aggiunti, in termini insiemistici, solo i nuovi token
        vocabolario = vocabolario.union(token)
    # Viene calcolata la lunghezza del vocabolario
    lunghezzaVocabolario = len(vocabolario)   
    # Viene restituita la lunghezza del vocabolario
    return lunghezzaVocabolario

# La funzione HapaxDistribution prende in input una lista di tokens e restituisce la distribuzione degli hapax
def HapaxDistribution(tokens):
    hapaxList = []
    # Viene calcolata la frequenza dei tokens
    freqdist = nltk.FreqDist(tokens)
    # Vengono calcolati gli hapax
    hapaxList = freqdist.hapaxes()
    # Vengono ottenuti il numero di segmenti col quale dividere il testo
    lenSplitIndex = math.ceil(len(tokens)/1000)
    # Viene ottenuta una lista per poter dividere il testo
    splitList = [i*1000 for i in range(1, lenSplitIndex+1)]
    # Viene inizializzata una lista per contare il numero di hapax per ogni segmento di testo
    hapaxDistr = [0 for _ in range(lenSplitIndex)]
    # Il testo viene spezzetato in segmenti
    temp = zip(chain([0], splitList), chain(splitList, [None])) 
    testoSegmentato = list(tokens[i : j] for i, j in temp) 
    # Si scorre tutto il testo diviso in segmenti utilizzando un indice e la relativa porzione di testo
    for indice, porzioneTesto in enumerate(testoSegmentato):
        # Per ogni hapax che è stato trovato precedentemente
        for hapax in hapaxList:
            # Viene verificato che l'hapax sia nella porzione di testo presa in considerazione
            if hapax in porzioneTesto:
                # Viene incrementato il numero di hapax della porzione di testo indicizzata da indice
                hapaxDistr[indice]+=1
    # Vengono restituisci gli hapax
    return hapaxDistr

# La funzione RapportoSostantiviVerbi prende in input le posTokens e restituisce il rapporto tra sostantivi e verbi
def RapportoSostantiviVerbi(posTokens):
    # Viene creata una lista di sole part of speech rimuovendo i tokens
    posTags = [posToken[1] for posToken in posTokens]
    # Viene calcolata la lista di frequenze delle part of speech
    freqdist = nltk.FreqDist(posTags)
    # Viene calcolata la frequenza dei sostantivi
    numSostantivi = freqdist['NN']+freqdist['NNS']+freqdist['NNP']+freqdist['NNPS']
    # Viene calcolata la frequenza dei verbi
    numVerbi = freqdist['VB']+freqdist['VBD']+freqdist['VBG']+freqdist['VBN']+freqdist['VBP']+freqdist['VBZ']
    # Viene restituito il rapporto tra sostantivi e verbi
    return (numSostantivi/numVerbi)

# La funzione FrequenzaPos prende in input le posTokens e restituisce le più frequenti con la loro frequenza 
def FrequenzaPos(posTokens):
    # Viene creata una lista di sole part of speech rimuovendo i tokens
    posTags = [posToken[1] for posToken in posTokens]
    # Viene calcolata la lista di frequenze delle part of speech
    tag = nltk.FreqDist(posTags)
    # Vengono restituite le 10 part of speech più frequenti e la loro frequenza
    return tag.most_common(10)

# La funzione ProbCondPOS prende in input le posTokens e tokenList e restituisce la probabilità condizionata
def ProbCondPOS(posTokens, tokensList):
    # Viene creata una lista di sole part of speech rimuovendo i tokens
    posTags = [posToken[1] for posToken in posTokens]
    # I bigrammi generati dalla funzione bigrams vengono trasformati in lista
    posTokensBigram = list(nltk.bigrams(posTags))
    # Viene creato un insieme di bigrammi di POS per calcolare tutti i bigrammi diversi
    bigrammiDiversi = set(posTokensBigram)
    listaBigrammaProb = []
    # Per ogni bigramma contenuto in bigrammiDiversi
    for bigramma in bigrammiDiversi:
        # Viene contato quante volte bigramma occorre in posTokensBigram
        frequenzaBigramma = posTokensBigram.count(bigramma)
        # Viene contato quante volte il POS di sinistra occorre in posTags
        frequenzaTesto = posTags.count(bigramma[0])
        # Viene calcolata la probabilità condizionata
        probCond = frequenzaBigramma*1.0/frequenzaTesto*1.0
        # Vengono aggiunti alla listaBigrammaProb il bigramma e la probCond
        listaBigrammaProb.append((bigramma,probCond))
    # Viene ordinata la lista
    listaOrdinata = sorted(listaBigrammaProb, key=lambda coppia: coppia[1], reverse=True)
    # Viene restituita la probabilità massima dei primi 10 bigrammi
    return listaOrdinata[0:9]    

# La funzione ForzaAssociativaMassima prende in input le posTokens e ne restituisce la forza associativa in termini di LMI
def ForzaAssociativaMassima(posTokens):
    # Viene creata una lista di sole part of speech rimuovendo i tokens
    posTags = [token[1] for token in posTokens]
    # I bigrammi generati dalla funzione bigrams vengono trasformati in lista
    posTokensBigram = list(nltk.bigrams(posTags))
    # Viene creato un insieme di bigrammi di POS per calcolare tutti i bigrammi diversi
    bigrammiDiversi = set(posTokensBigram)
    listaBigrammaAss = []
    # Per ogni bigramma contenuto in bigrammiDiversi
    for bigramma in bigrammiDiversi:
        # Viene contato quante volte bigramma occorre in posTokensBigram
        frequenzaBigramma = posTokensBigram.count(bigramma)
        # Calcolo la probabilità del bigramma
        probBigramma = frequenzaBigramma/len(posTokensBigram)
        posSinistra = bigramma[0]
        # Viene calcolato quante volte la posSinistra occorre in posTags
        frequenzaPosS = posTags.count(posSinistra)
        # Viene calcolata la probabilità della POS Sinistra
        probSinistra = frequenzaPosS/len(posTags)
        posDestra = bigramma[1]
        # Viene calcolato quante volte la posDestra occorre in posTags
        frequenzaPosD = posTags.count(posDestra)
        # Viene calcolata la probabilità della POS Destra
        probDestra = frequenzaPosD/len(posTags)
        # Viene calcolata la LMI
        LMI = frequenzaBigramma*math.log2(probBigramma/(probSinistra*probDestra))
        listaBigrammaAss.append((bigramma,LMI))
    # Viene ordinata la lista
    listaOrdinata = sorted(listaBigrammaAss, key=lambda coppia: coppia[1], reverse=True)
    # Viene restituita la forza associativa massima dei primi 10 bigrammi
    return listaOrdinata[0:9]


def main(files):
    # Verifica che siano presenti i parametri
    if len(files) < 1:
        print("Inserire i file come parametri")
        return
    # Rimozione del nome del programma dalla lista dei file
    files.pop(0)
    for file in files:
        fileInput = codecs.open(file, "r", "utf-8")
        raw = fileInput.read()
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        frasi = sent_tokenizer.tokenize(raw)
        list_tokens, list_posTokens = TextToList(frasi)
        lunghezza = CalcolaLunghezza(frasi)
        media = MediaCaratteri(frasi)
        vocabolario = GrandezzaVocabolario(frasi)
        hapax = HapaxDistribution(list_tokens)
        rapporto = RapportoSostantiviVerbi(list_posTokens)
        frequenza = FrequenzaPos(list_posTokens)
        probBigrammi = ProbCondPOS(list_posTokens, list_tokens)
        forzaAssociativa = ForzaAssociativaMassima(list_posTokens)

        # Stampo il numero totale delle frasi
        print ("Il file", (file), "contiene", len(frasi), "frasi")
        # Stampo il totale dei token
        print ("Il file",(file),"è lungo",(lunghezza),"token")
        # Stampo la lunghezza media delle frasi in termini di token
        print ("La lunghezza media delle frasi del", (file), "in termini di token è", (lunghezza/len(frasi)))
        # Stampo la lunghezza media dei caratteri delle parole
        print ('La lunghezza media dei caratteri delle parole del', (file), 'è', (media))
        # Stampo la grandezza del vocabolario
        print ('La grandezza del vocabolario del', (file), 'è', (vocabolario))
        # Stampo la distribuzione degli hapax all'aumentare del corpus per perzioni incrementali di 1000 token
        print ('La distribuzione degli hapax del', (file), 'è', (hapax))
        # Stampo il rapporto tra sostantivi e verbi
        print('Il rapporto tra sostantivi e verbi del', (file), 'è', (rapporto))  
        # Stampo la probabilità condizionata massima dei primi 10 bigrammi
        for coppia in probBigrammi:
            print ('La probabilità di', (coppia[0]), 'è di', (coppia[1]))
        # Stampo la forza associativa massima in termini di LMI dei primi 10 bigrammi
        for coppia in forzaAssociativa:
            print('La forza associativa di',(coppia[0]), 'è di', (coppia[1]))

main(sys.argv)
