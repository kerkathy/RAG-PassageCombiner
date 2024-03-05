# download prebuilt index
# The downloaded index will be in ~/.cache/pyserini/indexes/

prebuilt_index="wikipedia-dpr-100w.dpr-single-nq"
python -c "from pyserini.search import FaissSearcher; faiss_searcher = FaissSearcher.from_prebuilt_index('$prebuilt_index', 'facebook/dpr-question_encoder-single-nq-base')"
