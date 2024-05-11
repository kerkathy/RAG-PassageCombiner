# hotpotQA

flan_no_doc_exemplars = [
"""Q: Answer the following question.
Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
A: Walls and Bridges""",

"""Q: Answer the following question.
What is known as the Kingdom and has National Route 13 stretching towards its border?
A: Cambodia""",
 
"""Q: Answer the following question.
Jeremy Theobald and Christopher Nolan share what profession?
A: producer""",

"""Q: Answer the following question.
What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
A: The Phantom Hour""",

"""Q: Answer the following question.
Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: Scott Glenn""",

"""Q: Answer the following question.
Which car, produced by Ferrari from 1962 to 1964 for homologation into the FIA's Group 3 Grand Touring Car category inspired the Vandenbrink GTO?
A: Ferrari 250 GTO""",

"""Q: Answer the following question.
The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: Assante""",
]

# TODO
flan_one_doc_exemplars = [
]

flan_two_doc_exemplars = [
]


flan_three_docs_exemplars = [
"""Document:
Walls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.
"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".
Unfinished Music No. 1: Two Virgins is the first of three experimental albums released by John Lennon and Yoko Ono on Apple Records. It was the result of an all-night session of musical experimentation with Yoko in John's home studio at Kenwood, while his wife, Cynthia Lennon, was on holiday in Greece. Their debut recording is known not only for its avant-garde content, but also for its cover which features the couple naked: This made the album controversial to both the public and the parent record company EMI, which refused to distribute it. In an attempt to avoid controversy, the LP record was sold in a brown paper bag, and distributed by Track and Tetragrammaton in the United Kingdom and the United States respectively. "Two Virgins", while failing to chart in the UK, reached number 124 in the US. The album was followed six months later by "".

Q: Answer the following question.
Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
A: Walls and Bridges""",

"""Document:
Indonesian National Route 13 is a road in the national route system, and its course is entirely within the borders of the DKI Jakarta province. It's 15 kilometre path goes alongside with Java's arterial highway route, the Indonesian National Route 1, and is also indirectly connected to another route, that being the Indonesian National Route 2.
Cambodia ( ; Khmer: កម្ពុជា , or Kampuchea ] ), officially known as the Kingdom of Cambodia (Khmer: ព្រះរាជាណាចក្រកម្ពុជា , "Preăh Réachéanachâk Kâmpŭchéa", ] ), is a country located in the southern portion of the Indochina Peninsula in Southeast Asia. It is 181035 km2 in area, bordered by Thailand to the northwest, Laos to the northeast, Vietnam to the east, and the Gulf of Thailand to the southwest.
National Route 13 (Vietnamese: "Quốc lộ 13" ) is a highway in southern Vietnam stretching from the northeastern outskirts of Ho Chi Minh City, the commercial centre of the country, towards the border to Cambodia. The highway starts around Thủ Đức on the northern outskirts of Ho Chi Minh City, once the site of the military academy of the Army of the Republic of Vietnam, and travels north through the provinces of Bình Dương and Bình Phước. The highway passes through the districts of Thuận An, Thủ Dầu Một town, Bến Cát, Chơn Thành, Đồng Phú, Bình Long, and Lộc Ninh.

Q: Answer the following question.
What is known as the Kingdom and has National Route 13 stretching towards its border?
A: Cambodia""",

"""Document:
Jeremy Theobald is a British actor best known for his portrayal of "The Young Man", the main character in Christopher Nolan's 1998 major picture debut "Following", and for which Theobald was also a producer, Filming was scheduled around their day jobs. Jonathan Romney, writing in the "New Statesman", noted that "Nolan and his cast are terrific finds: I wouldn't normally say this to struggling artists, but they might want to give up their day jobs."
Veterinary support personnel in Japan do not currently hold any official state recognition and are known under a variety of equivalent names. Credentialing is carried out by various private organizations. These organizations are the Japan Animal Health Technicians Association (JAHTA), the Japanese Animal Hospital Association (JAHA), the Japan Small Animal Veterinary Association (JSAVA), the All Japan Veterinary Co-operative (JVC), and the Japanese Society of Animal Nursing. The Japanese Veterinary Nurses & Technicians Association (JVNTA), a non-certifying body which closed its doors in 2007, was one of the original member organizations of the IVNTA. In 2009 the Japanese Veterinary Nursing Association (JVNA) was organized as an effort to unify and standardize the profession in Japan and to seek state recognition. The JVNA, which has the support of the Japanese Veterinary Medical Association (JVMA—the national organization for veterinarians), may serve only as a temporary vehicle towards a single permanent national certifying body. All of the organizations have been collaborating since 2010 as the Council for Veterinary Nursing Examination and have reached an agreement to share a common examination in February 2012.
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter. He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.

Q: Answer the following question.
Jeremy Theobald and Christopher Nolan share what profession?
A: producer""",

"""Document:
The Burning Soil (German: "Der brennende Acker" ) is a 1922 German silent film directed by F.W. Murnau. It was made the same year as Murnau's "Nosferatu" and released in Germany around the same time. The film follows tells the story of a struggle over a plot of petroleum-rich land.
Nosferatu, eine Symphonie des Grauens (translated as Nosferatu: A Symphony of Horror; or simply Nosferatu) is a 1922 German Expressionist horror film, directed by F. W. Murnau, starring Max Schreck as the vampire Count Orlok. The film, shot in 1921 and released in 1922, was an unauthorized adaptation of Bram Stoker's "Dracula" (1897). Various names and other details were changed from the novel: for instance, "vampire" became "Nosferatu" and "Count Dracula" became "Count Orlok".
The Phantom Hour is a 2016 short film written and directed by Brian Patrick Butler. It officially premiered September 8, 2016 at the Horrible imaginings Film Festival in San Diego, California. The film was inspired by German Expressionist films such as Nosferatu and The Cabinet of Dr. Caligari.

Q: Answer the following question.
What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
A: The Phantom Hour""",

"""Document:
The Right Stuff is a 1979 book by Tom Wolfe about the pilots engaged in U.S. postwar research with experimental rocket-powered, high-speed aircraft as well as documenting the stories of the first Project Mercury astronauts selected for the NASA space program. "The Right Stuff" is based on extensive research by Wolfe, who interviewed test pilots, the astronauts and their wives, among others. The story contrasts the "Mercury Seven" and their families with test pilots such as Chuck Yeager, who was considered by many contemporaries as the best of them all, but who was never selected as an astronaut.
Theodore Scott Glenn (born January 26, 1941), better known as Scott Glenn, is an American actor. His roles have included Wes Hightower in "Urban Cowboy" (1980), astronaut Alan Shepard in "The Right Stuff" (1983), Emmett in "Silverado" (1985), Commander Bart Mancuso in "The Hunt for Red October" (1990), Jack Crawford in "The Silence of the Lambs" (1991), Roger in "Training Day" (2001), Ezra Kramer in "The Bourne Ultimatum" (2007), Kevin Garvey, Sr. in "The Leftovers" (2014–2017) and as Stick in both "Daredevil" (2015–) and "The Defenders" (2017).
Vertical Limit is a 2000 American survival thriller film directed by Martin Campbell and written by Robert King. The film stars Chris O'Donnell, Bill Paxton, Robin Tunney and Scott Glenn. The film was released on December 8, 2000 in the United States by Columbia Pictures, receiving mixed reviews from critics and grossing $215 million worldwide.

Q: Answer the following question.
Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: Scott Glenn""",

"""Document:
Group 5 was an FIA motor racing classification which was applied to four distinct categories during the years 1966 to 1982. Initially Group 5 regulations defined a Special Touring Car category and from 1970 to 1971 the classification was applied to limited production Sports Cars restricted to 5 litre engine capacity. The Group 5 Sports Car category was redefined in 1972 to exclude the minimum production requirement and limit engine capacity to 3 litres. From 1976 to 1982 Group 5 was for Special Production Cars, a liberal silhouette formula based on homologated production vehicles.
The Ferrari 250 GTO is a GT car produced by Ferrari from 1962 to 1964 for homologation into the FIA's Group 3 Grand Touring Car category. It was powered by Ferrari's Tipo 168/62 V12 engine.
The Vandenbrink GTO is a limited re-bodied version of the Ferrari 599 GTB Fiorano. This means an entirely new coachwork, designed by Michiel van den Brink of Vandenbrink Design, is fitted on the stripped chassis and drivetrain of a Ferrari 599 GTB production car. The car's styling is inspired by the 1962 Ferrari 250 GTO as a tribute.

Q: Answer the following question.
Which car, produced by Ferrari from 1962 to 1964 for homologation into the FIA's Group 3 Grand Touring Car category inspired the Vandenbrink GTO?
A: Ferrari 250 GTO""",
]

