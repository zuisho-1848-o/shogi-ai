/-
京都将棋は先手必勝（初期局面で手番の先手が勝つ）

証明の流れ:
  1. 証明DAG（Fairy-Stockfish に先手の手を選ばせて作った）から作った証明書（Data.lean、base64）を読み込む
  2. `check cert = true` を計算で確かめる（native_decide。Lean のコンパイラ／インタプリタを信頼する）
  3. 健全性の定理 `sound`（Cert.lean、Lean のカーネルで検査済み）から、証明書の0番目の局面＝初期局面が負けと分かる
-/
import Kyoto.Cert
import Kyoto.Data

namespace Kyoto

def b64val (c : UInt8) : Nat :=
  if 65 ≤ c && c ≤ 90 then (c - 65).toNat
  else if 97 ≤ c && c ≤ 122 then (c - 97 + 26).toNat
  else if 48 ≤ c && c ≤ 57 then (c - 48 + 52).toNat
  else if c == 43 then 62
  else if c == 47 then 63
  else 0

/-- base64 を復号する（末尾の '=' による詰め物に対応） -/
def decodeB64 (s : String) : ByteArray := Id.run do
  let src := s.toUTF8
  let mut out := ByteArray.emptyWithCapacity (src.size * 3 / 4)
  let mut i := 0
  while i + 3 < src.size do
    let a := b64val src[i]!
    let b := b64val src[i + 1]!
    let c := b64val src[i + 2]!
    let d := b64val src[i + 3]!
    let n := (a <<< 18) ||| (b <<< 12) ||| (c <<< 6) ||| d
    out := out.push (n >>> 16 &&& 255).toUInt8
    if src[i + 2]! != 61 then out := out.push (n >>> 8 &&& 255).toUInt8
    if src[i + 3]! != 61 then out := out.push (n &&& 255).toUInt8
    i := i + 4
  return out

def readLE (b : ByteArray) (off len : Nat) : Nat :=
  (List.range len).foldl (fun acc t => acc ||| (b[off + t]!.toNat <<< (8 * t))) 0

/-- 証明書の形式: u32 n、n × [key 18バイト, u8 kind, u16 rank, u32 子の数 c, c × u32 子の番号] -/
def parseCert (b : ByteArray) : Cert := Id.run do
  let n := readLE b 0 4
  let mut off := 4
  let mut key : Array Key := Array.emptyWithCapacity n
  let mut kind : Array Nat := Array.emptyWithCapacity n
  let mut rank : Array Nat := Array.emptyWithCapacity n
  let mut kids : Array (Array Nat) := Array.emptyWithCapacity n
  for _ in [0:n] do
    key := key.push (readLE b off 18)
    kind := kind.push (readLE b (off + 18) 1)
    rank := rank.push (readLE b (off + 19) 2)
    let c := readLE b (off + 21) 4
    off := off + 25
    let mut ks : Array Nat := Array.emptyWithCapacity c
    for t in [0:c] do
      ks := ks.push (readLE b (off + 4 * t) 4)
    kids := kids.push ks
    off := off + 4 * c
  return { key, kind, rank, kids }

def cert : Cert := parseCert (decodeB64 Data.all)

theorem cert_ok : check cert = true := by native_decide

theorem cert_head : 0 < cert.size ∧ cert.key[0]! = initKey ∧ cert.kind[0]! = 1 := by native_decide

/-- 京都将棋は先手必勝: 初期局面（先手番）で、手番側（先手）は必ず勝てる -/
theorem first_player_wins : Wins initKey := by
  obtain ⟨hpos, hkey, hkind⟩ := cert_head
  have h := (sound cert cert_ok 0 hpos).2 (by omega)
  rw [hkey] at h
  exact h

end Kyoto
