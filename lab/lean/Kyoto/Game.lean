/-
勝ち負けの定義（有限の手数で決着がつく、ゲーム理論的な意味での必勝・必敗）

  Wins k  : 手番側が必ず勝てる
  Loses k : 手番側が（相手が最善を尽くせば）必ず負ける

どちらも帰納的に定義する（＝有限の手数で終わる勝ち筋だけを数える）。
勝つ側は有限の手数で勝ち切るので、負ける側が千日手で逃げる余地はない（千日手の規則に左右されない）。
合法手が1つもない局面は、定義から手番側の負け（京都将棋のルールどおり）。
-/
import Kyoto.Rules

namespace Kyoto

mutual
  inductive Wins : Key → Prop
    | imm {k : Key} : immWin k = true → Wins k
    | step {k q : Key} : q ∈ moves k → Loses q → Wins k

  inductive Loses : Key → Prop
    | all {k : Key} : immWin k = false → (∀ q, q ∈ moves k → Wins q) → Loses k
end

end Kyoto
