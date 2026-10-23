/-
京都将棋のルール（計算できる関数として。lab/kyoto/rules.py と同じ定義・同じ指し手の順。Fairy-Stockfish の kyotoshogi と一致することを確認済み）

  盤 5×5。マス番号 = (段-1)×5 + 筋（0..4）。段1 が先手の陣。先手の前方は段が増える向き
  駒の種類: 0 玉 / 1 香 / 2 銀 / 3 桂 / 4 歩 / 5 と(香の裏) / 6 角(銀の裏) / 7 金(桂の裏) / 8 飛(歩の裏)
  盤の値 = 0（空き）または 1 + 種類 + 9×持ち主（持ち主 0 先手 / 1 後手）
  持ち駒: 持ち主ごとに組（0 香・1 銀・2 桂・3 歩）の枚数
  玉以外は動かすたびに必ず裏返る。取った駒は組として持ち駒になり、表でも裏でも空いているマスに打てる
  自玉に王手がかかる手は指せない。合法手がなければ負け

局面の key（自然数）: 盤 25マス × 5bit、持ち駒 8欄 × 2bit（bit 125 から）、手番 1bit（bit 141）
-/
namespace Kyoto

abbrev Key := Nat

structure Pos where
  b : Array Nat      -- 25マス
  hand : Array Nat   -- [先手の香,銀,桂,歩, 後手の香,銀,桂,歩]
  side : Nat         -- 0 先手番 / 1 後手番

def decode (k : Key) : Pos :=
  { b := (List.range 25).toArray.map (fun i => (k >>> (5 * i)) % 32),
    hand := (List.range 8).toArray.map (fun j => (k >>> (125 + 2 * j)) % 4),
    side := (k >>> 141) % 2 }

def encode (p : Pos) : Key :=
  let bb := (List.range 25).foldl (fun acc i => acc + (p.b[i]! <<< (5 * i))) 0
  let hh := (List.range 8).foldl (fun acc j => acc + (p.hand[j]! <<< (125 + 2 * j))) bb
  hh + (p.side <<< 141)

def ptype (v : Nat) : Nat := (v - 1) % 9
def owner (v : Nat) : Nat := (v - 1) / 9
def pcode (t o : Nat) : Nat := 1 + t + 9 * o

/-- 裏返した種類（玉はそのまま） -/
def flipT : Nat → Nat
  | 1 => 5 | 5 => 1 | 2 => 6 | 6 => 2 | 3 => 7 | 7 => 3 | 4 => 8 | 8 => 4 | t => t

def pairOf : Nat → Nat
  | 1 => 0 | 5 => 0 | 2 => 1 | 6 => 1 | 3 => 2 | 7 => 2 | _ => 3

def goldSteps : List (Int × Int) := [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)]

def steps : Nat → List (Int × Int)
  | 0 => [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
  | 2 => [(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)]
  | 4 => [(1, 0)]
  | 5 => goldSteps
  | 7 => goldSteps
  | 3 => [(2, -1), (2, 1)]
  | _ => []

def slides : Nat → List (Int × Int)
  | 1 => [(1, 0)]
  | 6 => [(1, 1), (1, -1), (-1, 1), (-1, -1)]
  | 8 => [(1, 0), (-1, 0), (0, 1), (0, -1)]
  | _ => []

def orient (d : Int × Int) (o : Nat) : Int × Int := if o == 0 then d else (-d.1, -d.2)

def onBoard (r f : Int) : Bool := 0 ≤ r && r < 5 && 0 ≤ f && f < 5
def sqOf (r f : Int) : Nat := (r * 5 + f).toNat

/-- 駒（マス i、値 v）の行き先の並び（rules.py の pseudo_moves と同じ順: 一歩ずつの向き → 走る向き） -/
def targets (b : Array Nat) (i v : Nat) : List Nat :=
  let t := ptype v
  let o := owner v
  let r : Int := i / 5
  let f : Int := i % 5
  let st := (steps t).filterMap fun d =>
    let (dr, df) := orient d o
    if onBoard (r + dr) (f + df) then some (sqOf (r + dr) (f + df)) else none
  let sl := (slides t).flatMap fun d =>
    let (dr, df) := orient d o
    -- 盤の端か駒に当たるまで（当たったマスも含む）。最大4歩
    let rec go (k : Nat) (fuel : Nat) : List Nat :=
      match fuel with
      | 0 => []
      | fuel + 1 =>
        let rr := r + dr * k
        let ff := f + df * k
        if !onBoard rr ff then []
        else
          let s := sqOf rr ff
          if b[s]! != 0 then [s] else s :: go (k + 1) fuel
    go 1 4
  st ++ sl

/-- by 側の駒が sq に利いているか -/
def attacks (b : Array Nat) (sq by_ : Nat) : Bool :=
  (List.range 25).any fun i =>
    let v := b[i]!
    v != 0 && owner v == by_ && (targets b i v).contains sq

def kingSq (b : Array Nat) (o : Nat) : Nat := (b.findIdx? (· == pcode 0 o)).getD 0

/-- 盤上の駒を動かす手（自玉の安全は未確認）。玉を取る手は含めない -/
def boardMoves (p : Pos) : List Pos :=
  let me := p.side
  (List.range 25).flatMap fun i =>
    let v := p.b[i]!
    if v == 0 || owner v != me then [] else
      (targets p.b i v).filterMap fun s =>
        let cap := p.b[s]!
        if cap != 0 && owner cap == me then none
        else if cap != 0 && ptype cap == 0 then none
        else
          let t := ptype v
          let nt := if t == 0 then 0 else flipT t
          let b' := (p.b.set! i 0).set! s (pcode nt me)
          let h' := if cap == 0 then p.hand else
            let j := 4 * me + pairOf (ptype cap)
            p.hand.set! j (p.hand[j]! + 1)
          some { b := b', hand := h', side := 1 - me }

def faces : Nat → List Nat
  | 0 => [1, 5] | 1 => [2, 6] | 2 => [3, 7] | _ => [4, 8]

/-- 持ち駒を打つ手（組 0..3、表・裏、マス 0..24 の順） -/
def drops (p : Pos) : List Pos :=
  let me := p.side
  (List.range 4).flatMap fun pr =>
    let j := 4 * me + pr
    if p.hand[j]! == 0 then [] else
      (faces pr).flatMap fun face =>
        (List.range 25).filterMap fun s =>
          if p.b[s]! != 0 then none
          else some { b := p.b.set! s (pcode face me), hand := p.hand.set! j (p.hand[j]! - 1), side := 1 - me }

/-- 合法手の後の局面（自玉に王手がかからないもの） -/
def moves (k : Key) : List Key :=
  let p := decode k
  ((boardMoves p ++ drops p).filter fun q => !attacks q.b (kingSq q.b p.side) (1 - p.side)).map encode

/-- 京都将棋には即勝ちの規則がない（勝ちは相手に合法手がなくなること） -/
def immWin (_ : Key) : Bool := false

/-- 初期局面（先手番）: 先手 1段目に と・銀・玉・金・歩、後手 5段目に 歩・金・玉・銀・と（後手から見て左から と 銀 玉 金 歩） -/
def initKey : Key := encode
  { b := #[pcode 5 0, pcode 2 0, pcode 0 0, pcode 7 0, pcode 4 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           pcode 4 1, pcode 7 1, pcode 0 1, pcode 2 1, pcode 5 1],
    hand := #[0, 0, 0, 0, 0, 0, 0, 0], side := 0 }

end Kyoto
