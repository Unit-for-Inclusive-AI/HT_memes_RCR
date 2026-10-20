# Chinese Persona Council Statistics (Stage 1: Offline Pruning)

This document details the **Core Council of Personas** selected offline from the raw 300 Chinese candidate pool (`artifacts/persona_pool_zh.json`) for the Anti-LGBTQ+ HT Meme classification pipeline.

---

## 1. Selection & Council Overview

- **Input Candidates Evaluated**: 291 valid profiles (from 300 raw personas)
- **Final Council Size**: **8 personas**
- **MMR Balance Factor (λ)**: `0.65` (65% quality, 35% non-redundancy)
- **Formality Balance**: 口语化: 2, 书面表达: 6

### Council Roster & Performance Metrics

| Order | Persona ID | User Hash | Archetype | Formality | Substance | Relevance | Base Score | MMR Score |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| #1 | `P_ZH_0199` | `fa20f08ec029` | Critical / 批判保守者 | `口语化` | 0.8440 | 0.9412 | 0.8926 | 0.8926 |
| #2 | `P_ZH_0045` | `799b31c30157` | Skeptical / 怀疑调侃者 | `书面表达` | 1.0000 | 0.7555 | 0.8778 | 0.5706 |
| #3 | `P_ZH_0017` | `dc1730964044` | Neutral Observer / 客观旁观者 | `口语化` | 0.7240 | 1.0000 | 0.8620 | 0.5603 |
| #4 | `P_ZH_0244` | `c97e91bebe1d` | Educator / 事实科普与理性分析者 | `书面表达` | 1.0000 | 0.6963 | 0.8481 | 0.5513 |
| #5 | `P_ZH_0146` | `47f65da55df1` | Cultural / 传统价值观持有者 | `书面表达` | 0.8000 | 0.6233 | 0.7117 | 0.4605 |
| #6 | `P_ZH_0030` | `accc5dbfcc18` | Supportive / 包容友善支持者 | `书面表达` | 1.0000 | 0.3455 | 0.6727 | 0.4360 |
| #7 | `P_ZH_0188` | `7937c0a54220` | Cultural / 传统价值观持有者 | `书面表达` | 1.0000 | 0.4548 | 0.7274 | 0.4597 |
| #8 | `P_ZH_0109` | `0a3a243449b5` | Critical / 批判保守者 | `书面表达` | 1.0000 | 0.3860 | 0.6930 | 0.4462 |

---

## 2. Archetype Representation

The selection enforced coverage across all 6 sociological archetypes to ensure balanced debate tension during the multi-agent Reflect-Critique-Refine (RCR) rounds:

| Archetype | Council Count | Role in Hate Meme Deliberation |
| :--- | :---: | :--- |
| **Supportive / 包容友善支持者** | 1 | Monitored minority dignity, anti-LGBTQ+ harassment, and inclusive standards |
| **Critical / 批判保守者** | 2 | Critical lens on progressive shifts, focused on social order and boundaries |
| **Cultural / 传统价值观持有者** | 2 | Grounded in classical cultural etiquette, traditional marriage, and family |
| **Skeptical / 怀疑调侃者** | 1 | Online satirist; probes for cynicism, parody, and distinguishes jokes from malice |
| **Educator / 事实科普与理性分析者** | 1 | Factual explainer, analyzes literal semantics, logic, and meme origin templates |
| **Neutral Observer / 客观旁观者** | 1 | Surface-level common-sense interpretation from an everyday netizen |

---

## 3. Individual Persona Profiles & Historical Evidence

### #1 — Persona `P_ZH_0199`
- **Archetype**: **Critical / 批判保守者**
- **User Hash**: `fa20f08ec029`
- **Selection Reason**: Archetype representation (Critical / 批判保守者)
- **Scores**: Base Quality: `0.8926` (Substance: `0.8440`, Relevance: `0.9412`) | MMR: `0.8926`
- **Writing Habits**: 简短精炼；喜用表情包。 (Formality: `口语化`, Avg chars: `18.5`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"铭记历史"*
  - *"吾辈自强"*
  - *"宝贝六一快乐[亲亲]"*
  - *"姐姐六一快乐🙂 昨天的杂志尊嘟是太米了🥰 谁能拒绝这么米的杂志呢！ 迫不及待想现在就收到漂亮杂志 ◔.̮◔✧！😉"*

### #2 — Persona `P_ZH_0045`
- **Archetype**: **Skeptical / 怀疑调侃者**
- **User Hash**: `799b31c30157`
- **Selection Reason**: Archetype representation (Skeptical / 怀疑调侃者)
- **Scores**: Base Quality: `0.8778` (Substance: `1.0000`, Relevance: `0.7555`) | MMR: `0.5706`
- **Writing Habits**: 表达详实；喜用表情包。 (Formality: `书面表达`, Avg chars: `26.2`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"基本上目前的电视剧尤其是古装剧从来不看，除了一张脸，啥也不是，古代不是动不动就接吻，就你侬我侬，中国的传统文化几乎丢完了，没有了中国人的骨子里的底蕴。"*
  - *"鲜花鲜肉不会说话"*
  - *"大家都要平安！❤"*
  - *"温暖[心][心][心][心]"*

### #3 — Persona `P_ZH_0017`
- **Archetype**: **Neutral Observer / 客观旁观者**
- **User Hash**: `dc1730964044`
- **Selection Reason**: Archetype representation (Neutral Observer / 客观旁观者)
- **Scores**: Base Quality: `0.8620` (Substance: `0.7240`, Relevance: `1.0000`) | MMR: `0.5603`
- **Writing Habits**: 简短精炼；喜用表情包。 (Formality: `口语化`, Avg chars: `13.5`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"哈哈哈，好搞笑[笑cry][笑cry]"*
  - *"太逗了哈哈哈哈哈"*
  - *"平台也负有一定责任吧"*
  - *"真是吃的太饱了！应该重判以儆效尤。"*

### #4 — Persona `P_ZH_0244`
- **Archetype**: **Educator / 事实科普与理性分析者**
- **User Hash**: `c97e91bebe1d`
- **Selection Reason**: Archetype representation (Educator / 事实科普与理性分析者)
- **Scores**: Base Quality: `0.8481` (Substance: `1.0000`, Relevance: `0.6963`) | MMR: `0.5513`
- **Writing Habits**: 表达详实；语气强烈；喜用表情包。 (Formality: `书面表达`, Avg chars: `48.8`, Emoji: `True`, Punctuation: `True`)
- **Historical Comments Mined from Source**:
  - *"宝宝[抱一抱]"*
  - *"🍊💙你知道吗？昨天下午我被人骂了，她们说我往楼下滴水，把阳台上晾的衣服都打湿了，只有我知道那不是水，是我趴在阳台上想你流的眼泪"*
  - *"🍊💙因友情而心生厌恶？因爱情而心生憎恶？姜黎非身世终成谜！💕青丘孤女鞠婧祎姜黎非 与同窗们的修行之路隐藏多少谜踪？邀您一同揭秘🤘🏼"*
  - *"𝙝𝙪𝙜 𝙢𝙚 𝙝𝙪𝙜 𝙢𝙚 𝘽𝙖𝙗𝙮 𝙟𝙪𝙨𝙩 𝙝𝙪𝙜 𝙢𝙚 𝙝𝙪𝙜 𝙢𝙚每一次心动的原因，全部都是因为你"*

### #5 — Persona `P_ZH_0146`
- **Archetype**: **Cultural / 传统价值观持有者**
- **User Hash**: `47f65da55df1`
- **Selection Reason**: Archetype representation (Cultural / 传统价值观持有者)
- **Scores**: Base Quality: `0.7117` (Substance: `0.8000`, Relevance: `0.6233`) | MMR: `0.4605`
- **Writing Habits**: 表达详实；语气强烈。 (Formality: `书面表达`, Avg chars: `35.5`, Emoji: `False`, Punctuation: `True`)
- **Historical Comments Mined from Source**:
  - *"别再让你那恶心的前夫上热搜了行不，大家都不想看见他，行行好，污染心灵和眼睛。他不应该再蹭你的名气"*
  - *"圣依，你应该是何惟芳，而不是胜意！！！！切记！"*

### #6 — Persona `P_ZH_0030`
- **Archetype**: **Supportive / 包容友善支持者**
- **User Hash**: `accc5dbfcc18`
- **Selection Reason**: Archetype representation (Supportive / 包容友善支持者)
- **Scores**: Base Quality: `0.6727` (Substance: `1.0000`, Relevance: `0.3455`) | MMR: `0.4360`
- **Writing Habits**: 表达详实；喜用表情包。 (Formality: `书面表达`, Avg chars: `29.2`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"致敬所有缉毒英雄！"*
  - *"一定要确保安全啊"*
  - *"2025平安顺遂"*
  - *"#周冬雨[超话]##可惜我是水瓶座# 周冬雨祝大家“元旦快乐”，BGM放的却是周杰伦的《黑色幽默》，嗯，可能就是随心情选的，是我熟悉的水瓶座，祝这个爱笑的女人2025大展宏图~[努力]"*

### #7 — Persona `P_ZH_0188`
- **Archetype**: **Cultural / 传统价值观持有者**
- **User Hash**: `7937c0a54220`
- **Selection Reason**: High composite MMR & stylometric complementarity
- **Scores**: Base Quality: `0.7274` (Substance: `1.0000`, Relevance: `0.4548`) | MMR: `0.4597`
- **Writing Habits**: 表达详实；喜用表情包。 (Formality: `书面表达`, Avg chars: `27.5`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"历史不容遗忘，和平需要捍卫！"*
  - *"铭记历史 吾辈自强[加油]"*
  - *"点染双唇宣言，书写自信格调，沉淀先锋态度，演绎百变真我，和#M·A·C魅可品牌光彩代言人# 一同探索色彩科学，解锁妆容无限可能。[羞嗒嗒]"*
  - *"[打call]好棒"*

### #8 — Persona `P_ZH_0109`
- **Archetype**: **Critical / 批判保守者**
- **User Hash**: `0a3a243449b5`
- **Selection Reason**: High composite MMR & stylometric complementarity
- **Scores**: Base Quality: `0.6930` (Substance: `1.0000`, Relevance: `0.3860`) | MMR: `0.4462`
- **Writing Habits**: 表达详实；喜用表情包。 (Formality: `书面表达`, Avg chars: `28.5`, Emoji: `True`, Punctuation: `False`)
- **Historical Comments Mined from Source**:
  - *"特别理解这种感受[泪][泪][泪][泪][泪][泪]"*
  - *"依依好会写，感受到了内心的细腻"*
  - *"想起了同样的场景载我老爹[心]老爸一直说姑娘你慢着点[哈哈][哈哈][哈哈]下车还和朋友说：这孩子开车太猛了😂😂"*
  - *"[抱一抱][抱一抱]开起来一定很爽"*

---

## 4. Next Step: Stage 2 Integration

During runtime evaluation in [`Code/meme_persona_pipeline.py`](../Code/meme_persona_pipeline.py), the online meme selector will condition on the specific meme text and visual summary to dynamically select the **Top-3 complementary debaters** from this 8-member council.