# 🤖 Tehisintellekti rakendamise projektiplaani mall (CRISP-DM)

<br>
<br>


## 🔴 1. Äritegevuse mõistmine
*Fookus: mis on probleem ja milline on hea tulemus?*


### 🔴 1.1 Kasutaja kirjeldus ja eesmärgid
Kellel on probleem ja miks see lahendamist vajab? Mis on lahenduse oodatud kasu? Milline on hetkel eksisteeriv lahendus?

> Probleem on see, et tudeng ei tea, millised õppeained olemas on, ja ei oska kuskilt otsida. ÕISis ei näe eelmiste semestrite aineid, osad on peidus. Kasuks on see, et leiad kiiresti seda, mida sa otsid ja säästad aega. Sa ei leia aineid, millest tegelikult oleksid väga huvitatud. ÕISis on robustne lahendus, kus saab otsida ainult aine nime ja mõnede filtrite põhjal. Kui sa ei tea, mida otsida, siis on raske midagi leida. 

### 🔴 1.2 Edukuse mõõdikud
Kuidas mõõdame rakenduse edukust? Mida peab rakendus teha suutma?

> Rakendus leiab vastavalt kirjeldusele top 5 kõige lähedasemat vastet. Programm on edukas kui vastus leitakse mõistliku kiirusega ja info on relevantne. Hea näidik oleks kui võetakse rohkem valikaineid või instituudiväliseid aineid. Tagasisidet saaks koguda. Testcase'id testimiseks. 

### 🔴 1.3 Ressursid ja piirangud
Millised on ressursipiirangud (nt aeg, eelarve, tööjõud, arvutusvõimsus)? Millised on tehnilised ja juriidilised piirangud (GDPR, turvanõuded, platvorm)? Millised on piirangud tasuliste tehisintellekti mudelite kasutamisele?

> Peamised piirangud on arvutusvõimsus ja arenduse aeg. Oluline küsimus on andmekaitse vestluse sisu poolest. Makseinfo peab olema konfidentsiaalne. 

<br>
<br>


## 🟠 2. Andmete mõistmine
*Fookus: millised on meie andmed?*

### 🟠 2.1 Andmevajadus ja andmeallikad
Milliseid andmeid (ning kui palju) on lahenduse toimimiseks vaja? Kust andmed pärinevad ja kas on tagatud andmetele ligipääs?

> Andmed on olemas ja pärinevad ÕISist. Oleks vaja kindlasti paari viimase semesti ainete andmeid. Ligipääs on tagatud, see on avalik informatsioon. Andmed on algselt töötlemata. Kui eeldada, et meie programm töötab eesti keeles, siis on vaja ainult eestikeelseid andmeid. Kindlast on vaja ainete põhilist infot (nimi, kood, EAP, kirjeldus, asukoht, eeldusained, hindamine jpm.).

### 🟠 2.2 Andmete kasutuspiirangud
Kas andmete kasutamine (sh ärilisel eesmärgil) on lubatud? Kas andmestik sisaldab tundlikku informatsiooni?

> Andmete kasutamine on avalik informatsioon, aga sisaldab võimalikke isikuandmeid, seega ei ole hetkeseisuga ärieesmärkidel kasutatav. 

### 🟠 2.3 Andmete kvaliteet ja maht
Millises formaadis andmeid hoiustatakse? Mis on andmete maht ja andmestiku suurus? Kas andmete kvaliteet on piisav (struktureeritus, puhtus, andmete kogus) või on vaja märkimisväärset eeltööd?

> Andmed on kraabitud ja on csv formaadis, mittetöötlemata. Algne andmekvaliteet ei ole kindlast piisav. Andmed vajavad märkimisväärset eeltööd. Algandmeid on palju, umbes 45.3MB. 223x3031.

### 🟠 2.4 Andmete kirjeldamise vajadus
Milliseid samme on vaja teha, et kirjeldada olemasolevaid andmeid ja nende kvaliteeti.

> Teha andmeanalüüsi ja uurida väljade sisu ja formaati.

<br>
<br>


## 🟡 3. Andmete ettevalmistamine
Fookus: Toordokumentide viimine tehisintellekti jaoks sobivasse formaati.

### 🟡 3.1 Puhastamise strateegia
Milliseid samme on vaja teha andmete puhastamiseks ja standardiseerimiseks? Kui suur on ettevalmistusele kuluv aja- või rahaline ressurss?

> Andmetest on vaja eemaldada NaN väärtused, neid pole vaja. Tuleb otsusta mis keeles andmeid kasutada. Osad väljad on sisulise poole pealt identsed. Object (Sõne) tüüpi väärtused sisaldavad reavahetusmärke. Rahaline ressurss põhimõtteliselt puudub, ajaline on märkimisväärne. 

### 🟡 3.2 Tehisintellektispetsiifiline ettevalmistus
Kuidas andmed tehisintellekti mudelile sobivaks tehakse (nt tükeldamine, vektoriseerimine, metaandmete lisamine)?

> Andmetöötlusest peaks piisama. 

<br>
<br>

## 🟢 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 🟢 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> Esiteks on vaja vektoresituse mudelit. Vaja on LLMi mudelit, mis genereerib RAGist saadud info põhjal vastuse. KOmponentide rakendamine toimub üksteise järel.  

### 🟢 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> Hetkel kasutab embedimiseks ****** ja LLM on google/gemma-****** Kasutame valmis teenust API. 

### 🟢 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> Kas vastused on need, mida ootame. Kui tahame saada matemaatika õppeaineid, siis saame matemtika õppeaineid ikka. Ühesõnaga vigade analüüs. Lisame ka testjuhud, kus on eeldetermineeritud promptid ja ainekoodid, mis peaksid välja tulema. 

### 🟢 4.4 Rakenduse arendus
Milliste sammude abil on plaanis/on võimalik rakendust järk-järgult parandada (viibadisain, erinevte mudelite testimine jne)?

> Kõigepealt on vaja saavutada süsteemi töötavus täies mahus ja vastavus eesmärkidele. Selle järel tuleb veebiarendus ja muud sarnased "deployment" sarnased arendusetapid. Nende sammude käigus on parandamiseks oma sammud. Süsteemi põhjalik testimine on esimese etapi juures suurim fookus. Edasise kasutajani jõudmise juures on alguses testimine ja hiljem kasutajate tagasiside.


### 🟢 4.5 Riskijuhtimine
Kuidas maandatakse tehisintellektispetsiifilisi riske (hallutsinatsioonid, kallutatus, turvalisus)?

> Etteantud kitsendused LLMi jaoks. Sisendite piiramine vahepeal. Piirame väljundit. 

<br>
<br>

## 🔵 5. Tulemuste hindamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🔵 5.1 Vastavus eesmärkidele
Kuidas hinnata, kas rakendus vastab seatud eesmärkidele?

> Läbi testide saaks osade eesmärkide saavutatust hinnata. Võimalik oleks ka test-kasutajate tagasiside ja hinnangud. 

<br>
<br>

## 🟣 6. Juurutamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🟣 6.1 Integratsioon
Kuidas ja millise liidese kaudu lõppkasutaja rakendust kasutab? Kuidas rakendus olemasolevasse töövoogu integreeritakse (juhul kui see on vajalik)?

> Lõppkasutaja kasutaks seda omaette veebilehena. võimalik läbi arvuti ja läpka. Telefonivaate implementeerimine on mingis mõttes ka triviaalne. 

### 🟣 6.2 Rakenduse elutsükkel ja hooldus
Kes vastutab süsteemi tööshoidmise ja jooksvate kulude eest? Kuidas toimub rakenduse uuendamine tulevikus?

> Süsteemi kulud hetkel on mudeli maksumus ja veebilehe maksumus. Nende kulude eest võiks vastutada sama isik, suuremas pildis on tegemist raamatupidamisprobleemiga, aga tegelikult saaks sellega tegeleda "peamine arendaja". 