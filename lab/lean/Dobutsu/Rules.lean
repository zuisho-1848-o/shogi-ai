/-
どうぶつしょうぎのルール（計算できる関数として）

局面は「手番側から見た向き」の 64bit の数（lab/dobutsu/solve.c と同じ形式。左右反転の正規化はしない）:
  盤 12マス × 4bit（マス番号 i = 段×3 + 筋、段0 が手番側から見て奥＝相手の陣の奥）
    駒コード: 0 空き / 1 ライオン / 2 キリン / 3 ぞう / 4 ひよこ / 5 にわとり。相手の駒は +8
  持ち駒: bit 48 から 2bit ずつ 手番側のキリン・ぞう・ひよこ、bit 54 から 相手のキリン・ぞう・ひよこ
指した後の局面は、相手から見た向きに回転して表す（盤を180度回し、先後を入れ替える）。

勝ち負けの規則（solve.c と同じ）:
  - 相手のライオンを取れる → その手で勝ち
  - 自分のライオンを奥の段に進め、そこに相手の利きがない（トライ成立）→ その手で勝ち
  - 自分のライオンが既に奥の段にいる（相手がトライを止めなかった）→ 勝ち
  これらの「即勝ち」の手は `moves` に含めず、`immWin` で表す。持ち駒はどこにでも打てる（ひよこを奥の段に打ってもよい）。
-/
namespace Dobutsu

abbrev Key := UInt64

def cell (k : Key) (i : Nat) : Nat := ((k >>> (4 * i).toUInt64) &&& 15).toNat
def handAt (k : Key) (j : Nat) : Nat := ((k >>> (48 + 2 * j).toUInt64) &&& 3).toNat

/-- 盤（12マス）と持ち駒（手番側3種・相手3種） -/
structure Pos where
  b : Array Nat
  hm : Array Nat
  ho : Array Nat

def decode (k : Key) : Pos :=
  { b := (List.range 12).toArray.map (cell k), hm := #[handAt k 0, handAt k 1, handAt k 2], ho := #[handAt k 3, handAt k 4, handAt k 5] }

def encode (p : Pos) : Key :=
  let board := (List.range 12).foldl (fun acc i => acc ||| ((p.b[i]!).toUInt64 <<< (4 * i).toUInt64)) 0
  let h1 := (List.range 3).foldl (fun acc j => acc ||| ((p.hm[j]!).toUInt64 <<< (48 + 2 * j).toUInt64)) board
  (List.range 3).foldl (fun acc j => acc ||| ((p.ho[j]!).toUInt64 <<< (54 + 2 * j).toUInt64)) h1

/-- 駒の動ける向き（段の差, 筋の差）。手番側の駒について。段が減る向きが前 -/
def dirs : Nat → List (Int × Int)
  | 1 => [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
  | 2 => [(-1, 0), (1, 0), (0, -1), (0, 1)]
  | 3 => [(-1, -1), (-1, 1), (1, -1), (1, 1)]
  | 4 => [(-1, 0)]
  | 5 => [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
  | _ => []

/-- 相手の駒（コード ≥ 8）が マス s に利いているか（相手の駒は前後が逆） -/
def oppAttacks (b : Array Nat) (s : Nat) : Bool :=
  let sr : Int := s / 3
  let sc : Int := s % 3
  (List.range 12).any fun i =>
    let v := b[i]!
    v ≥ 8 && (dirs (v - 8)).any fun (dr, dc) => (i / 3 : Int) - dr == sr && (i % 3 : Int) + dc == sc

/-- 指した後の局面を相手から見た向きに回転して key にする -/
def flip (p : Pos) : Key :=
  encode { b := (List.range 12).toArray.map (fun i => let v := p.b[11 - i]!; if v == 0 then 0 else v ^^^ 8),
           hm := p.ho, ho := p.hm }

/-- 盤上の駒を動かす手の結果。`none` は指せない手、`some (true, _)` は即勝ちの手、`some (false, q)` は普通の手 -/
def boardMove (p : Pos) (i : Nat) (dr dc : Int) : Option (Bool × Key) :=
  let v := p.b[i]!
  let nr : Int := (i / 3 : Int) + dr
  let nc : Int := (i % 3 : Int) + dc
  if nr < 0 || nr > 3 || nc < 0 || nc > 2 then none
  else
    let s := (nr * 3 + nc).toNat
    let cap := p.b[s]!
    if cap != 0 && cap < 8 then none           -- 自分の駒がある
    else if cap == 9 then some (true, 0)        -- 相手のライオンを取る
    else
      let piece := if v == 4 && nr == 0 then 5 else v   -- ひよこは奥の段で成る
      let b' := (p.b.set! i 0).set! s piece
      let hm' := if cap == 0 then p.hm else
        let t := cap - 8
        let j := (if t == 5 then 4 else t) - 2      -- 取った駒を持ち駒に（にわとりはひよこに戻る）
        p.hm.set! j (p.hm[j]! + 1)
      if v == 1 && nr == 0 && !oppAttacks b' s then some (true, 0)   -- トライ成立
      else some (false, flip { b := b', hm := hm', ho := p.ho })

/-- 盤上の駒を動かす手（solve.c と同じ順: マス 0..11、向きは dirs の順）の結果の並び -/
def boardResults (p : Pos) : List (Bool × Key) :=
  (List.range 12).flatMap fun i =>
    let v := p.b[i]!
    if v == 0 || v ≥ 8 then [] else (dirs v).filterMap fun (dr, dc) => boardMove p i dr dc

/-- 持ち駒を打つ手（持ち駒の種類 0..2、マス 0..11 の順） -/
def drops (p : Pos) : List Key :=
  (List.range 3).flatMap fun j =>
    if p.hm[j]! == 0 then [] else
      (List.range 12).filterMap fun s =>
        if p.b[s]! != 0 then none
        else some (flip { b := p.b.set! s (2 + j), hm := p.hm.set! j (p.hm[j]! - 1), ho := p.ho })

/-- 即勝ちがあるか（自ライオンが既に奥の段にいる / 相手ライオンを取れる / トライが成立する） -/
def immWin (k : Key) : Bool :=
  let p := decode k
  (List.range 3).any (fun i => p.b[i]! == 1) || (boardResults p).any (·.1)

/-- 即勝ちではない合法手の後の局面（相手から見た向き）。自ライオンが既に奥の段にいる局面では空 -/
def moves (k : Key) : List Key :=
  let p := decode k
  if (List.range 3).any (fun i => p.b[i]! == 1) then []
  else ((boardResults p).filter (!·.1)).map (·.2) ++ drops p

/-- 初期局面（先手番、先手から見た向き） -/
def initKey : Key := encode
  { b := #[11, 9, 10, 0, 12, 0, 0, 4, 0, 2, 1, 3], hm := #[0, 0, 0], ho := #[0, 0, 0] }

end Dobutsu
