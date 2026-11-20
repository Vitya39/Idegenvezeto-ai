import sqlite3

def create_and_populate_db(db_path: str = "tourism.db") -> None:
    """Létrehozza és feltölti az SQLite adatbázist az alapértelmezett turisztikai helyszínekkel."""
    # Kapcsolódás az adatbázishoz
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tábla létrehozása
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS places (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        opening_hours TEXT NOT NULL,
        ticket_price_huf INTEGER NOT NULL,
        description TEXT NOT NULL
    )
    ''')


    places_data = [
        # --- BUDAPEST ---
        (
            "parlament",
            "Országház",
            "budapest",
            47.507157,
            19.045765,
            "08:00-18:00",
            6500,
            "A magyar Országház Budapest egyik legikonikusabb nevezetessége a Duna partján."
        ),
        (
            "halaszbastya",
            "Halászbástya",
            "budapest",
            47.501372,
            19.034873,
            "09:00-23:00",
            1200,
            "Neoromán stílusú kilátóterasz a Várhegyen, csodálatos panorámával."
        ),
        (
            "szent_istvan_bazilika",
            "Szent István-bazilika",
            "budapest",
            47.500833,
            19.053611,
            "09:00-17:45",
            3200,
            "Budapest egyik legnagyobb temploma, ahonnan kilátás nyílik a belvárosra."
        ),
        (
            "szechenyi_lanchid",
            "Széchenyi Lánchíd",
            "budapest",
            47.498118,
            19.042514,
            "00:00-24:00",
            0,
            "Magyarország első állandó kőhídja a Duna felett."
        ),
        (
            "szechenyi_gyogyfurdo",
            "Széchenyi Gyógyfürdő",
            "budapest",
            47.518889,
            19.082222,
            "07:00-20:00",
            10500,
            "Európa egyik legnagyobb fürdőkomplexuma a Városligetben."
        ),

        # --- OLASZORSZÁG  ---
        (
            "trevi_kut",
            "Trevi-kút",
            "olaszország",
            41.9009,
            12.4833,
            "00:00-24:00",
            0,
            "Róma legnagyobb és egyik leghíresebb barokk szökőkútja."
        ),

        (
            "vatikan",
            "Szent Péter-bazilika (Vatikán)",
            "olaszország",
            41.9022,
            12.4539,
            "07:00-19:00",
            0,
            "A katolikus egyház központja, reneszánsz építészeti remekmű."
        ),
        (
            "castel_sant_angelo",
            "Angyalvár (Castel Sant'Angelo)",
            "olaszország",
            41.9031,
            12.4663,
            "09:00-19:30",
            6000,
            "Henger alakú építmény, amely Hadrianus császár mauzóleumaként épült."
        ),
        (
            "pantheon",
            "Pantheon",
            "olaszország",
            41.8986,
            12.4769,
            "09:00-19:00",
            2000,
            "Ókori római templom, híres a kupoláján lévő nyitott szemnyílásról (oculus)."
        ),

        # --- PÁRIZS ---
        (
            "eiffel_torony",
            "Eiffel-torony",
            "párizs",
            48.8584,
            2.2945,
            "09:00-00:00",
            11000,
            "Párizs legikonikusabb rácsos vaskonstrukciójú tornya a Mars-mezőn."
        ),
        (
            "louvre_muzeum",
            "Louvre Múzeum",
            "párizs",
            48.8606,
            2.3376,
            "09:00-18:00",
            9000,
            "A világ legnagyobb művészeti múzeuma, itt látható a Mona Lisa is."
        ),
        (
            "notre_dame",
            "Notre-Dame katedrális",
            "párizs",
            48.8530,
            2.3499,
            "08:00-19:00",
            0,
            "Gótikus stílusú katolikus katedrális Párizs szívében, a Île de la Cité-n."
        ),
        (
            "diadaliv",
            "Diadalív (Arc de Triomphe)",
            "párizs",
            48.8738,
            2.2950,
            "10:00-23:00",
            5500,
            "A francia nemzeti öntudat híres emlékműve a Champs-Élysées végén."
        ),
        (
            "sainte_chapelle",
            "Sainte-Chapelle",
            "párizs",
            48.8554,
            2.3450,
            "09:00-19:00",
            4500,
            "Gótikus királyi kápolna, lenyűgöző ólomüveg ablakairól híres."
        ),

        # --- A VILÁG 7 CSODÁJA ---
        (
            "kina_nagy_fal",
            "Kína nagy fala",
            "vilagcsoda",
            40.4319,
            116.5704,
            "07:30-17:30",
            15000,
            "Ősi erődrendszer sorozat Kínában, a világ leghosszabb ember alkotta építménye."
        ),
        (
            "chichen_itza",
            "Chichén Itzá",
            "vilagcsoda",
            20.6843,
            -88.5678,
            "08:00-17:00",
            12000,
            "Maja romváros Mexikóban, híres a Kukulkán-piramisról."
        ),
        (
            "petra",
            "Petra",
            "vilagcsoda",
            30.3285,
            35.4444,
            "06:00-18:00",
            25000,
            "Sziklába vájt ókori nabateus város Jordániában."
        ),
        (
            "machu_picchu",
            "Machu Picchu",
            "vilagcsoda",
            -13.1631, -72.5450,
            "06:00-17:30",
            30000,
            "15. századi inka város az Andokban, Peru területén."
        ),
        (
            "megvalt_krisztus",
            "A Megváltó Krisztus szobra",
            "vilagcsoda",
            -22.9519, -43.2105,
            "08:00-19:00",
            8000,
            "Art Deco stílusú hatalmas Jézus-szobor Rio de Janeiróban."
        ),
        (
            "colosseum_csoda",
            "Colosseum (Világcsoda)",
            "vilagcsoda",
            41.8902,
            12.4922,
            "08:30-19:00",
            6800,
            "Az ókori Róma jelképe (kategóriája szerint a világcsodák közé is sorolják)."
        ),
        (
            "taj_mahal",
            "Taj Mahal",
            "vilagcsoda",
            27.1751,
            78.0421,
            "06:00-19:00",
            6000,
            "Fehér márványból épült csodálatos mauzóleum Agrában, Indiában."
        )
    ]

    cursor.executemany('''
    INSERT OR IGNORE INTO places (id, name, category, lat, lon, opening_hours, ticket_price_huf, description)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', places_data)

    conn.commit()
    print("Az adatbázis sikeresen frissült!\n")


    cursor.execute("SELECT category, COUNT(*) FROM places GROUP BY category")
    summary = cursor.fetchall()
    print("Helyszínek száma kategóriánként az adatbázisban:")
    for row in summary:
        print(f"- {row[0].upper()}: {row[1]} db helyszín")
    conn.close()

if __name__ == "__main__":
    create_and_populate_db()
