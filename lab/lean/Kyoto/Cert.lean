/-
証明書と、その検査器・健全性の定理

証明書（Python の lab/kyoto/cert_kyoto.py が証明DAGから作る）は局面の並びで、各局面に
  key（手番側から見た局面）, kind（0=負け 1=勝ち 2=即勝ち）, rank（終局までの手数）, kids（子の番号の並び）
を持つ。検査器 `checkNode` は各局面について次を確かめる:
  kind = 2: immWin key（即勝ちがある）
  kind = 1: 子が1つで、その子は負け（kind 0）、rank が小さく、その局面が moves key に含まれる
  kind = 0: 即勝ちがなく、moves key がちょうど子の局面の並びに一致し、どの子も勝ち（kind ≠ 0）で rank が小さい
健全性: 全局面が検査に通れば、kind 0 の局面は Loses、それ以外は Wins（rank についての帰納法）。
-/
import Kyoto.Game

namespace Kyoto

structure Cert where
  key : Array Key
  kind : Array Nat
  rank : Array Nat
  kids : Array (Array Nat)

def Cert.size (c : Cert) : Nat := c.key.size

/-- kind 1（勝ち）の検査: 子が1つで、負け局面で、rank が小さく、合法手で行ける -/
def checkWin (c : Cert) (i : Nat) : Bool :=
  c.kids[i]!.size == 1 && decide (c.kids[i]![0]! < c.size) && c.kind[c.kids[i]![0]!]! == 0 &&
    decide (c.rank[c.kids[i]![0]!]! < c.rank[i]!) && (moves c.key[i]!).contains c.key[c.kids[i]![0]!]!

/-- kind 0（負け）の子についての検査 -/
def kidOk (c : Cert) (r : Nat) (j : Nat) : Bool :=
  decide (j < c.size) && c.kind[j]! != 0 && decide (c.rank[j]! < r)

/-- kind 0（負け）の検査: 即勝ちがなく、合法手の行き先がちょうど子の並びで、どの子も勝ち局面で rank が小さい -/
def checkLose (c : Cert) (i : Nat) : Bool :=
  !immWin c.key[i]! && moves c.key[i]! == c.kids[i]!.toList.map (fun j => c.key[j]!) &&
    c.kids[i]!.all (kidOk c c.rank[i]!)

def checkNode (c : Cert) (i : Nat) : Bool :=
  if c.kind[i]! == 2 then immWin c.key[i]!
  else if c.kind[i]! == 1 then checkWin c i
  else if c.kind[i]! == 0 then checkLose c i
  else false

def check (c : Cert) : Bool :=
  c.kind.size == c.size && c.rank.size == c.size && c.kids.size == c.size &&
    (List.range c.size).all (checkNode c)

theorem checkWin_spec {c : Cert} {i : Nat} (h : checkWin c i = true) :
    c.kids[i]![0]! < c.size ∧ c.kind[c.kids[i]![0]!]! = 0 ∧ c.rank[c.kids[i]![0]!]! < c.rank[i]! ∧
      c.key[c.kids[i]![0]!]! ∈ moves c.key[i]! := by
  simp only [checkWin, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.contains_iff_mem] at h
  exact ⟨h.1.1.1.2, h.1.1.2, h.1.2, h.2⟩

theorem checkLose_spec {c : Cert} {i : Nat} (h : checkLose c i = true) :
    immWin c.key[i]! = false ∧ moves c.key[i]! = c.kids[i]!.toList.map (fun j => c.key[j]!) ∧
      ∀ j ∈ c.kids[i]!, j < c.size ∧ c.kind[j]! ≠ 0 ∧ c.rank[j]! < c.rank[i]! := by
  simp only [checkLose, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, Array.all_eq_true_iff_forall_mem] at h
  refine ⟨h.1.1, h.1.2, fun j hj => ?_⟩
  have := h.2 j hj
  simp only [kidOk, Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq] at this
  exact ⟨this.1.1, this.1.2, this.2⟩

/-- 検査に通った証明書の各局面について、kind 0 なら負け、それ以外なら勝ち -/
theorem sound (c : Cert) (hc : check c = true) :
    ∀ i, i < c.size → (c.kind[i]! = 0 → Loses c.key[i]!) ∧ (c.kind[i]! ≠ 0 → Wins c.key[i]!) := by
  have hall : ∀ i, i < c.size → checkNode c i = true := by
    intro i hi
    simp only [check, Bool.and_eq_true, List.all_eq_true, List.mem_range] at hc
    exact hc.2 i hi
  suffices h : ∀ r, ∀ i, i < c.size → c.rank[i]! = r →
      (c.kind[i]! = 0 → Loses c.key[i]!) ∧ (c.kind[i]! ≠ 0 → Wins c.key[i]!) by
    intro i hi
    exact h _ i hi rfl
  intro r
  induction r using Nat.strongRecOn with
  | _ r ih =>
    intro i hi hr
    have hn := hall i hi
    by_cases h2 : c.kind[i]! = 2
    · simp only [checkNode, h2] at hn
      exact ⟨fun h0 => by omega, fun _ => Wins.imm hn⟩
    by_cases h1 : c.kind[i]! = 1
    · simp only [checkNode, h1] at hn
      obtain ⟨hj, hk0, hrk, hmem⟩ := checkWin_spec hn
      refine ⟨fun h0 => by omega, fun _ => Wins.step hmem ?_⟩
      exact (ih _ (hr ▸ hrk) _ hj rfl).1 hk0
    by_cases h0 : c.kind[i]! = 0
    · simp only [checkNode, h0] at hn
      obtain ⟨himm, hmv, hkids⟩ := checkLose_spec hn
      refine ⟨fun _ => Loses.all himm ?_, fun h => absurd h0 h⟩
      intro q hq
      rw [hmv] at hq
      simp only [List.mem_map, Array.mem_toList_iff] at hq
      obtain ⟨j, hjmem, rfl⟩ := hq
      obtain ⟨hj, hknz, hrk⟩ := hkids j hjmem
      exact (ih _ (hr ▸ hrk) _ hj rfl).2 hknz
    · simp only [checkNode] at hn
      simp [h2, h1, h0] at hn

end Kyoto
