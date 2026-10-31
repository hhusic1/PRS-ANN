# Literatura

Radovi su grupisani kao folderi u `docs/`. Za svaki rad dati su puni navod, linkovi i kratko šta iz njega uzimamo za projekat. PDF-ovi u repou su lokalne kopije; za citiranje se koriste DOI ili arXiv linkovi.

## Zašto IVF (`zasto-ivf/`)

**H. Jégou, M. Douze, C. Schmid: Product Quantization for Nearest Neighbor Search.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), str. 117–128, 2011. DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57), [HAL](https://inria.hal.science/inria-00514462). PDF: [`zasto-ivf/jegou-product-quantization.pdf`](zasto-ivf/jegou-product-quantization.pdf)

Osnova algoritma. Grubi kvantizator se trenira k-meansom, svaki vektor ide u invertovanu listu svog centroida, a upit se dodjeljuje na više najbližih listi (naš `nprobe`). Iz ovog rada potiče i skup SIFT1M: 100.000 vektora za učenje, 1.000.000 u bazi i 10.000 upita, dimenzija 128. Za SIFT navode grubi kvantizator od 1.000 centroida naviše. Mi implementiramo samo IVF dio, bez PQ kompresije (IVF-Flat).

**J. Sivic, A. Zisserman: Video Google: A Text Retrieval Approach to Object Matching in Videos.** Proceedings of the 9th IEEE International Conference on Computer Vision (ICCV), vol. 2, str. 1470–1477, 2003. DOI: [10.1109/ICCV.2003.1238663](https://doi.org/10.1109/ICCV.2003.1238663). PDF: [`zasto-ivf/sivic-video-google.pdf`](zasto-ivf/sivic-video-google.pdf)

Porijeklo ideje. Deskriptori se vektorski kvantizuju u „vizuelne riječi“, a pretraga koristi invertovani fajl, kao kod pretrage teksta. Jégou et al. svoj grubi kvantizator opisuju kao isti pristup.

**A. Babenko, V. Lempitsky: The Inverted Multi-Index.** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), str. 3069–3076, 2012. DOI: [10.1109/CVPR.2012.6248038](https://doi.org/10.1109/CVPR.2012.6248038). PDF nije u repou.

Poznata alternativa invertovanom indeksu: vektor se dijeli na dvije polovine, svaka ima svoj kodebuk, a ćelije su Kartezijev proizvod ćelija dvije polovine. Spominjemo je kao alternativu koju ne implementiramo.

**D. Baranchuk, A. Babenko, Y. Malkov: Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors.** European Conference on Computer Vision (ECCV), Lecture Notes in Computer Science, str. 209–224, 2018. DOI: [10.1007/978-3-030-01258-8_13](https://doi.org/10.1007/978-3-030-01258-8_13), arXiv: [1802.02422](https://arxiv.org/abs/1802.02422). PDF: [`zasto-ivf/baranchuk-revisiting-inverted-indices.pdf`](zasto-ivf/baranchuk-revisiting-inverted-indices.pdf)

Argument za klasični invertovani indeks naspram multi-indeksa: većina regija multi-indeksa ostaje prazna, pa je efektivan broj regija mnogo manji od teoretskog.

## Različiti pristupi ANN-u (`razliciti-pristupi-ann/`)

**W. Li, Y. Zhang, Y. Sun, W. Wang, M. Li, W. Zhang, X. Lin: Approximate Nearest Neighbor Search on High Dimensional Data — Experiments, Analyses, and Improvement.** IEEE Transactions on Knowledge and Data Engineering, 32(8), str. 1475–1488, 2020. DOI: [10.1109/TKDE.2019.2909204](https://doi.org/10.1109/TKDE.2019.2909204), arXiv: [1610.02455](https://arxiv.org/abs/1610.02455) (u repou je verzija v1.0). PDF: [`razliciti-pristupi-ann/li-ann-experiments.pdf`](razliciti-pristupi-ann/li-ann-experiments.pdf)

Pregled i eksperimentalno poređenje velikog broja ANN metoda. Uzimamo metodologiju mjerenja: tačni kNN kao referenca i krive odnosa recall–brzina umjesto pojedinačnih tačaka.

**Y. A. Malkov, D. A. Yashunin: Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), str. 824–836, 2020. DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473), arXiv: [1603.09320](https://arxiv.org/abs/1603.09320). PDF: [`razliciti-pristupi-ann/malkov-hnsw.pdf`](razliciti-pristupi-ann/malkov-hnsw.pdf)

Predstavnik grafovskih metoda (HNSW). Pretraga je pohlepni obilazak grafa sa nepravilnim pristupom memoriji, što je slabije pogodno za SIMD od skeniranja kontinualnih listi kod IVF-a.

**M. Datar, N. Immorlica, P. Indyk, V. S. Mirrokni: Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.** Proceedings of the 20th Annual Symposium on Computational Geometry (SoCG), str. 253–262, 2004. DOI: [10.1145/997817.997857](https://doi.org/10.1145/997817.997857). PDF: [`razliciti-pristupi-ann/datar-lsh-p-stable.pdf`](razliciti-pristupi-ann/datar-lsh-p-stable.pdf)

Predstavnik metoda hashiranja (LSH) koji radi direktno u euklidskom prostoru.

**M. Muja, D. G. Lowe: Scalable Nearest Neighbor Algorithms for High Dimensional Data.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(11), str. 2227–2240, 2014. DOI: [10.1109/TPAMI.2014.2321376](https://doi.org/10.1109/TPAMI.2014.2321376). PDF: [`razliciti-pristupi-ann/muja-flann.pdf`](razliciti-pristupi-ann/muja-flann.pdf)

Predstavnik metoda zasnovanih na stablima (FLANN: randomizovana k-d stabla, prioritetno k-means stablo i automatski izbor algoritma). Sekcija 5.1 opisuje distribuiranu pretragu preko MPI-ja: podaci se ravnomjerno dijele na procese, a rezultati se spajaju.

## Paralelizacije (`paralelizacije/`)

**M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, H. Jégou: The Faiss Library.** arXiv: [2401.08281](https://arxiv.org/abs/2401.08281), 2024. PDF: [`paralelizacije/douze-faiss-library.pdf`](paralelizacije/douze-faiss-library.pdf)

FAISS je naša referenca za poređenje. FAISS paralelizuje preko OpenMP-a: batch upita se dijeli na niti, a pojedinačan upit se ne dijeli. To je isti izbor kao „paralelno po upitima“ u našem planu. Rad definiše i imbalance faktor listi, koji mjerimo i kod sebe.

**F. André, A.-M. Kermarrec, N. Le Scouarnec: Quicker ADC: Unlocking the Hidden Potential of Product Quantization With SIMD.** IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(5), str. 1666–1677, 2021. DOI: [10.1109/TPAMI.2019.2952606](https://doi.org/10.1109/TPAMI.2019.2952606), arXiv: [1812.09162](https://arxiv.org/abs/1812.09162). PDF: [`paralelizacije/andre-quicker-adc.pdf`](paralelizacije/andre-quicker-adc.pdf)

SIMD (AVX2, AVX-512, `pshufb`) za ubrzanje PQ pretrage. Na IVF-Flat nije direktno primjenjiv, jer ne koristimo PQ kodove, ali pokazuje kako se ANN pretraga prilagođava širini SIMD registara.

**J. Johnson, M. Douze, H. Jégou: Billion-Scale Similarity Search with GPUs.** IEEE Transactions on Big Data, 7(3), str. 535–547, 2021. DOI: [10.1109/TBDATA.2019.2921572](https://doi.org/10.1109/TBDATA.2019.2921572), arXiv: [1702.08734](https://arxiv.org/abs/1702.08734). PDF: [`paralelizacije/johnson-faiss-gpu.pdf`](paralelizacije/johnson-faiss-gpu.pdf)

GPU implementacija FAISS-a (IVFADC, izbor k najboljih u registrima: WarpSelect). Šira referenca, van obima CPU verzije.

**P. Wieschollek, O. Wang, A. Sorkine-Hornung, H. P. A. Lensch: Efficient Large-Scale Approximate Nearest Neighbor Search on the GPU.** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), str. 2027–2035, 2016. DOI: [10.1109/CVPR.2016.223](https://doi.org/10.1109/CVPR.2016.223), arXiv: [1702.05911](https://arxiv.org/abs/1702.05911). PDF: [`paralelizacije/wieschollek-ann-gpu.pdf`](paralelizacije/wieschollek-ann-gpu.pdf)

ANN pretraga na GPU-u, zasnovana na proširenju ideje Product Quantization. Šira referenca.
