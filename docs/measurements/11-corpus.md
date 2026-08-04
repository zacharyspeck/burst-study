
==============================================================================
CORPUS BUILT AND VERIFIED

Skylion007/openwebtext
revision 79d93d786212f7344586290adb811d4ae6a1762c

  held out        10,485,760 tokens      10,240 seq   heldout.bin
  training     2,499,805,184 tokens   2,441,216 seq   149 shards
  total        2,510,290,944 tokens  5,020,581,888 bytes

THE HELD-OUT SLICE IS AT THE FRONT. bursts/context.txt is openwebtext
document 73, and two arm texts are documents 104 and 193. Front-placed
they are held out; end-placed every one would have been trained on and
every step 10 metric would have measured memorisation while claiming
to measure representation.

DATA ORDER IS PER-SEED. The digests below are a CONTRACT: the training
loop must recompute its order and match one of them before consuming a
single batch. A recorded order that is never checked proves nothing.
==============================================================================

==============================================================================
SOURCE
==============================================================================
  dataset          Skylion007/openwebtext
  revision         79d93d786212f7344586290adb811d4ae6a1762c
  files used       25 of 80
  source bytes     7,572,655,845

  Files, not a stream: a stream has no seek so a resume rereads
  everything, and a stream of `main` is identified by nothing.
  Order comes from the numbered filenames, never a directory
  listing.

==============================================================================
GEOMETRY
==============================================================================
block            sequences          tokens           bytes
heldout             10,240      10,485,760      20,971,520
training         2,441,216   2,499,805,184   4,999,610,368
total            2,451,456   2,510,290,944   5,020,581,888

  149 shards x 16,384 sequences, exact -- no ragged final shard.
  dtype uint16 little-endian, raw, no header: filesize / 2 == token count exactly.

==============================================================================
VERIFICATION -- every number by more than one route
==============================================================================
  manifest sha256      e305bea71aa68c8ad355b7da3915125530b65a5de391f8dc08766b8b9dd5b255
  ^ carry this SEPARATELY from the manifest; a corrupted
    manifest would otherwise validate corrupted shards.
  blocks present       150 of 150
  blocks re-hashed     150
  boundaries checked   149 against arithmetic
  tokens from manifest 2,510,290,944
  tokens from disk     2,510,290,944
  training tokens      2,499,805,184 (expected 2,499,805,184)
  held-out disjoint    True

------------------------------------------------------------------------------
tokenizer identity -- a probe, not a version string
------------------------------------------------------------------------------
  probe                ran
  probe token sha256   e47ede6a3794902a8a86360e35bb26a823c128ba15a7f54b05d116efcfb34857
  agrees with build    True
  built by             transformers 5.14.1, tokenizers 0.22.2

==============================================================================
DATA ORDER -- per seed, and a CONTRACT
==============================================================================
  permutation over 2,441,216 training sequences,
  derived from SHA-256 in counter mode -- no library PRNG, so no
  library version can change it.

  seed  permutation sha256
     0  7e7121d15745824e5de6ed6ee65959782b8b60fba0a60968736760a48e8ba7b7
     1  cd9993e6c67ea75e5ecd630128880047336302a32eb4fec0910638b36fdb96cb
     2  417ba42248019f35c411c1de6fb855a1549d20c849a96e1c9aaec55462163872
     3  5dce4e667fcadd49dc5821e97a531b35e26b88109c1871e38d8fe258864e3ebd
     4  7a8f08bbed270f3101ca2067ba268d8aa15306e259589c88f1d9e89b76fc4ed0
     5  9e033f016de13f2ffb1672ed0b533d43e1c4b8f708af026a6659b51b43e9bb4f
     6  a369c1ceafaef3ca43c8c25fbf20af9e30fc2a8252289cbb5638b09e0acb2508
     7  3941d630ed209b1d4694525cbc0abbf26accb83fc811af87242afa78a0962127
     8  f12d537d483f0cbcf191538eb6192a029d722c762ae3b0c5beb01fbefda6fd05
     9  fba21ac6eb9d3cebeada7783e0e0aec54b51056a1de2370f911220352c5da03f

  THE TRAINING LOOP MUST CALL
  data_order.verify_permutation(seed, 2441216, <digest>)
  before consuming a batch. Measured at 1.98s. A run that serves a
  different order than its provenance claims is unreproducible and
  silently so. Cross-module obligation 4.

==============================================================================
PROVENANCE
==============================================================================
  Corpus: Skylion007/openwebtext at revision 79d93d786212f7344586290adb811d4ae6a1762c, 25 of 80 pinned Parquet files, 7,572,655,845 source bytes.
  Tokenized to 2,510,290,944 tokens in 150 blocks: a 10,485,760-token held-out slice and 149 training shards totalling 2,499,805,184 tokens.
  The training slice is exactly configs/base.yaml's expected_token_budget (2,499,805,184 = 256 x 1024 x 9536), so the loader's assertion holds.
  THE HELD-OUT SLICE IS AT THE FRONT, and that is load-bearing rather than cosmetic: it places the corpus-derived texts this repo already committed (bursts/context.txt is document 73, and two arm texts are documents 104 and 193) OUTSIDE the training slice.
  Placed at the end every one of them would have been trained on, and every step 10 metric would have been measuring memorisation while reporting representation.
  Disjointness is structural: the sampler permutes range(2,441,216), which indexes the training shards only, and the held-out slice is a separate file.
  assert_heldout_disjoint_from_training reports True.
  Every number was checked by more than one route: block hashes (150 re-hashed), file size against recorded token count, the total summed from the manifest (2,510,290,944) against the total summed from disk (2,510,290,944), and 149 shard boundaries against arithmetic.
  Data order is PER-SEED: 10 seeds, each a permutation of the same 2,441,216 training sequences derived from SHA-256 rather than any library PRNG.
  The digests below are what the training loop must check its own order against before consuming a batch -- cross-module obligation 4.
  Tokenizer drift is guarded by a probe rather than a version string: the committed passage tokenizes to e47ede6a3794902a here, matching what built the corpus (True).
  A tokenizer that would produce a different corpus produces a different probe hash whatever it calls itself.
  Verification found no problems.
