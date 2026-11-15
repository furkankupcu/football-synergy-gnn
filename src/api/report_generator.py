"""Takım performans raporu ve karşılaştırma raporu üretici."""

from __future__ import annotations
import tempfile, warnings
from pathlib import Path
from typing import Optional

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

from src import db
from src.analysis.insights import team_strengths_weaknesses, h2h_tactical_recommendations
from src.analysis.style import style_fingerprint as _sfp, cluster_teams as _cluster
from src.analysis.transfer import find_similar_players, squad_position_gaps

# ── Renk paleti ────────────────────────────────────────────────────────────
BG="#0d1117"; S1="#161b22"; S2="#21262d"; BR="#30363d"
TX="#e6edf3"; MU="#8b949e"
BL="#58a6ff"; GR="#7ee787"; OR="#ffa657"; RE="#ff7b72"; PU="#d2a8ff"; YE="#f8c97a"

RC = {
    "figure.facecolor":BG,"axes.facecolor":S1,"axes.edgecolor":BR,
    "axes.labelcolor":TX,"axes.titlecolor":TX,"xtick.color":MU,"ytick.color":MU,
    "text.color":TX,"grid.color":BR,"grid.alpha":0.4,"font.family":"DejaVu Sans","font.size":9,
}
def _rc(): plt.rcParams.update(RC)
def _pc(pos): return YE if pos=="G" else BL if pos=="D" else GR if pos=="M" else RE
def _sp(ax):
    for s in ax.spines.values(): s.set_edgecolor(BR); s.set_linewidth(1.0)

def _border(ax, color=BR, lw=1.2):
    """Axes'e görünür bir border çerçevesi çizer (axis("off") sonrası da çalışır)."""
    ax.add_patch(plt.Rectangle((0,0),1,1,transform=ax.transAxes,
                                fill=False,edgecolor=color,linewidth=lw,
                                clip_on=False,zorder=20))

def _cap(ax, text):
    """Panel içinin alt kenarına italik açıklama satırı ekler."""
    ax.text(0.01, 0.013, text, transform=ax.transAxes,
            fontsize=5.8, color=MU, ha="left", va="bottom",
            clip_on=True, style="italic", wrap=False,
            bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.75),
            zorder=25)

# ── Pitch çizimi (koordinat sistemi: x=derinlik 0-52, y=genişlik 0-100) ───
def _half_pitch(ax, bg=S1):
    ax.set_facecolor(bg); ax.set_xlim(-3,55); ax.set_ylim(-3,103)
    ax.set_aspect("equal"); ax.axis("off")
    kw=dict(color=BR,lw=1.2,zorder=1)
    # Ceza sahası büyük
    ax.plot([0,16.5],[21.1,21.1],**kw); ax.plot([0,16.5],[78.9,78.9],**kw)
    ax.plot([16.5,16.5],[21.1,78.9],**kw)
    # Ceza sahası küçük
    ax.plot([0,5.5],[36.8,36.8],**kw); ax.plot([0,5.5],[63.2,63.2],**kw)
    ax.plot([5.5,5.5],[36.8,63.2],**kw)
    # Kale
    ax.plot([0,-2.44],[45.2,45.2],color=BL,lw=2,zorder=2)
    ax.plot([0,-2.44],[54.8,54.8],color=BL,lw=2,zorder=2)
    ax.plot([-2.44,-2.44],[45.2,54.8],color=BL,lw=2,zorder=2)
    # Penaltı noktası + yayı
    ax.plot(11,50,"o",color=BR,ms=2.5,zorder=2)
    ax.add_patch(Arc((11,50),18.3,18.3,angle=0,theta1=-53,theta2=53,color=BR,lw=1.2))
    # Orta çizgi (yarı saha sınırı)
    ax.plot([52,52],[0,100],**kw)
    # Zemin dikdörtgen
    ax.add_patch(plt.Rectangle((0,0),52,100,fill=True,facecolor=bg,edgecolor=BR,lw=1.2,zorder=0))

def _goal_frame(ax):
    """Kale ağzı önden görünüm — şut & kurtarış yerleşim haritası.
    set_aspect kaldırıldı; hücreyi doldurarak okunaklı kalıyor."""
    ax.set_facecolor(S2)
    ax.set_xlim(33, 67)        # goal posts: 36.8 – 63.2  (±3 kenar boşluğu)
    ax.set_ylim(-0.6, 3.6)    # crossbar: 2.44m          (alt/üst boşluk)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(BR); s.set_linewidth(1.2)

    # Net arka planı (gol içi)
    ax.add_patch(plt.Rectangle((36.8,0),26.4,2.44,
                                facecolor="#0f1f0f",edgecolor="none",alpha=0.7,zorder=2))
    # Çim
    ax.add_patch(plt.Rectangle((33,-0.6),34,0.6,
                                facecolor="#1a2e1a",edgecolor="none",alpha=0.6,zorder=1))
    ax.axhline(0,color="#2e4a2e",lw=5,alpha=0.5,zorder=3)

    # Direkler + üst çıta
    kw=dict(color="#e0e0e0",lw=2.5,zorder=6)
    ax.plot([36.8,36.8],[0,2.44],**kw)
    ax.plot([63.2,63.2],[0,2.44],**kw)
    ax.plot([36.8,63.2],[2.44,2.44],**kw)

    # File ızgarası
    for x in np.linspace(36.8,63.2,8)[1:-1]:
        ax.plot([x,x],[0,2.44],color=BR,lw=0.5,alpha=0.45,zorder=4)
    for y in np.linspace(0,2.44,6)[1:-1]:
        ax.plot([36.8,63.2],[y,y],color=BR,lw=0.5,alpha=0.45,zorder=4)

    # Bölge etiketleri (sol / orta / sağ)
    for xi,lbl in [(39.5,"Sol"),(50,"Orta"),(60.5,"Sağ")]:
        ax.text(xi,-0.38,lbl,ha="center",va="top",fontsize=6,color=MU,zorder=7)
    # Yükseklik etiketleri
    for yi,lbl in [(0.4,"Alt"),(1.22,"Orta"),(2.1,"Üst")]:
        ax.text(34.2,yi,lbl,ha="right",va="center",fontsize=5.5,color=MU,zorder=7)

# ── Veri yükleyici ──────────────────────────────────────────────────────────
def _load(team:str, season:Optional[str]=None)->dict:
    slug=f"%{team}%"; p={"slug":slug}
    # sf = matches tablosu filtrelemesi (alias yok), sf2 = JOIN'li sorgular için
    sf  ="AND season=:season"   if season else ""
    sf2 ="AND m.season=:season" if season else ""
    if season: p["season"]=season

    matches=db.read_sql(f"""
        SELECT match_id,match_date,round,season,league,
               home_team_name,away_team_name,home_goals,away_goals,winner_code,
               CASE WHEN home_team_name ILIKE :slug THEN 'home' ELSE 'away' END AS side
        FROM matches
        WHERE (home_team_name ILIKE :slug OR away_team_name ILIKE :slug) {sf}
        ORDER BY match_date""",p)
    if matches.empty: return {}

    matches["kg"]=matches.apply(lambda r:r["home_goals"] if r["side"]=="home" else r["away_goals"],axis=1)
    matches["og"]=matches.apply(lambda r:r["away_goals"] if r["side"]=="home" else r["home_goals"],axis=1)
    matches["opp"]=matches.apply(lambda r:r["away_team_name"] if r["side"]=="home" else r["home_team_name"],axis=1)
    matches["res"]=matches.apply(lambda r:"G" if r["kg"]>r["og"] else("B" if r["kg"]==r["og"] else "M"),axis=1)

    shots=db.read_sql(f"""SELECT s.* FROM shots s
        JOIN matches m ON m.match_id=s.match_id
        WHERE (m.home_team_name ILIKE :slug OR m.away_team_name ILIKE :slug)
          AND ((s.is_home=true AND m.home_team_name ILIKE :slug)
            OR (s.is_home=false AND m.away_team_name ILIKE :slug)) {sf2}""",p)

    shots_ag=db.read_sql(f"""SELECT s.* FROM shots s
        JOIN matches m ON m.match_id=s.match_id
        WHERE (m.home_team_name ILIKE :slug OR m.away_team_name ILIKE :slug)
          AND ((s.is_home=true AND m.away_team_name ILIKE :slug)
            OR (s.is_home=false AND m.home_team_name ILIKE :slug)) {sf2}""",p)

    players=db.read_sql(f"""SELECT ps.player_name,ps.position,
        COUNT(DISTINCT ps.match_id) AS mac,ROUND(AVG(ps.rating)::numeric,2) AS rat,
        SUM(ps.goals) AS gol,SUM(ps.goal_assist) AS ast,
        ROUND(SUM(ps.expected_goals)::numeric,2) AS xg,
        ROUND(SUM(ps.expected_assists)::numeric,2) AS xa,
        SUM(ps.total_shots) AS sut,SUM(ps.minutes_played) AS dk,
        SUM(ps.total_pass) AS pas,SUM(ps.accurate_pass) AS ipas,
        SUM(ps.key_pass) AS kpas,SUM(ps.progressive_ball_carries_count) AS prun,
        SUM(ps.duel_won) AS dw,SUM(ps.duel_lost) AS dl,
        SUM(ps.interception_won) AS intp
        FROM player_stats ps
        JOIN matches m ON m.match_id=ps.match_id
        WHERE ps.team_name ILIKE :slug {sf2}
        GROUP BY ps.player_name,ps.position
        HAVING SUM(ps.minutes_played)>=90
        ORDER BY mac DESC,rat DESC NULLS LAST""",p)

    team_st=db.read_sql(f"""SELECT ts.* FROM team_stats ts
        JOIN matches m ON m.match_id=ts.match_id WHERE ts.period='ALL'
        AND ((ts.team_side='home' AND m.home_team_name ILIKE :slug)
          OR (ts.team_side='away' AND m.away_team_name ILIKE :slug)) {sf2}""",p)

    xgd=db.read_sql(f"""SELECT m.round,m.match_date,ts.expected_goals AS xg,
        CASE WHEN ts.team_side='home' THEN m.home_goals ELSE m.away_goals END AS gol
        FROM team_stats ts JOIN matches m ON m.match_id=ts.match_id
        WHERE ts.period='ALL' AND ((ts.team_side='home' AND m.home_team_name ILIKE :slug)
          OR (ts.team_side='away' AND m.away_team_name ILIKE :slug)) {sf2}
        ORDER BY m.match_date""",p)

    W=(matches["res"]=="G").sum(); D=(matches["res"]=="B").sum(); L=(matches["res"]=="M").sum()
    return dict(team_name=team,matches=matches,shots=shots,shots_ag=shots_ag,
                players=players,team_st=team_st,xgd=xgd,
                W=int(W),D=int(D),L=int(L),PTS=int(W*3+D),
                GF=int(matches["kg"].sum()),GA=int(matches["og"].sum()),
                TXG=round(float(shots["xg"].sum()) if not shots.empty else 0,1))

# ── Isı haritası (seaborn KDE + bölge analizi) ─────────────────────────────
def _heatmap(ax, df, title, colors, label_zones=True):
    _half_pitch(ax, bg=BG)
    ax.set_title(title, fontsize=9, color=TX, pad=5, loc="left")
    df=df.dropna(subset=["player_coord_x","player_coord_y"])
    if len(df)<5: return

    x=df["player_coord_x"].clip(0,50).values
    y=df["player_coord_y"].clip(0,100).values

    # Scipy KDE → contourf (seaborn kdeplot clip'i desteklemediğinde sorun çıkarabilir)
    xi=np.linspace(0,50,80); yi=np.linspace(0,100,80)
    Xi,Yi=np.meshgrid(xi,yi)
    try:
        k=gaussian_kde(np.vstack([x,y]),bw_method=0.25)
        Z=k(np.vstack([Xi.ravel(),Yi.ravel()])).reshape(Xi.shape)
        cmap=LinearSegmentedColormap.from_list("hm",[BG]+colors)
        ax.contourf(Xi,Yi,Z,levels=20,cmap=cmap,alpha=0.82,zorder=3)
    except Exception:
        pass

    if label_zones:
        # Bölge bazlı % hesapla
        zones=[
            ("Ceza\nSahası", (0,16.5),(21.1,78.9)),
            ("Merkez\nHücum",  (16.5,28),(30,70)),
            ("Uzak\nMesafe",   (28,52),(10,90)),
        ]
        for zlbl,(x0,x1),(y0,y1) in zones:
            mask=((df["player_coord_x"]>=x0)&(df["player_coord_x"]<x1)&
                  (df["player_coord_y"]>=y0)&(df["player_coord_y"]<=y1))
            pct=mask.sum()/max(len(df),1)*100
            cx=(x0+x1)/2; cy=(y0+y1)/2
            ax.text(cx,cy,f"{pct:.0f}%",ha="center",va="center",
                    fontsize=8,fontweight="bold",color="white",
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="#00000066",edgecolor="none"),
                    zorder=12)
            ax.text(cx,cy-5,zlbl,ha="center",va="center",fontsize=6,color=MU,zorder=12)

# ── Şut haritası ────────────────────────────────────────────────────────────
def _shotmap(ax, df, title, show_xg_size=True):
    _half_pitch(ax)
    ax.set_title(title, fontsize=9, color=TX, pad=5, loc="left")
    df=df.dropna(subset=["player_coord_x","player_coord_y"])
    if df.empty: return

    type_cfg={
        "goal":  dict(color=GR, base_s=160, alpha=0.95, marker="*", edge="white", zorder=12),
        "save":  dict(color=BL, base_s=45,  alpha=0.55, marker="o", edge="none",  zorder=8),
        "miss":  dict(color=OR, base_s=30,  alpha=0.45, marker="o", edge="none",  zorder=7),
        "block": dict(color=PU, base_s=30,  alpha=0.45, marker="s", edge="none",  zorder=7),
    }
    handles=[]
    for t,g in df.groupby("shot_type"):
        cfg=type_cfg.get(t,dict(color=MU,base_s=25,alpha=0.35,marker="o",edge="none",zorder=6))
        if show_xg_size and "xg" in g.columns:
            sizes=(g["xg"].fillna(0.05).clip(0.01,1)*cfg["base_s"]*3).values
        else:
            sizes=cfg["base_s"]
        ax.scatter(g["player_coord_x"],g["player_coord_y"],
                   c=cfg["color"],s=sizes,alpha=cfg["alpha"],
                   marker=cfg["marker"],edgecolors=cfg["edge"],lw=0.7,zorder=cfg["zorder"])
        handles.append(mpatches.Patch(color=cfg["color"],label=t,alpha=0.8))
    ax.legend(handles=handles,loc="upper right",facecolor=S2,edgecolor=BR,fontsize=6.5,
              framealpha=0.9,ncol=2)
    # xG toplamı — axes fraksiyonuyla konumlandır (legend ile çakışmaz)
    total_xg=df["xg"].sum() if "xg" in df.columns else 0
    goals_cnt=(df["shot_type"]=="goal").sum()
    ax.text(0.02,0.97,f"xG {total_xg:.1f} | {int(goals_cnt)} gol | {len(df)} şut",
            transform=ax.transAxes,fontsize=7,color=MU,ha="left",va="top",zorder=15)

# ── Auto insights ────────────────────────────────────────────────────────────
def _insights(d:dict)->list:
    shots,shots_ag,matches,players=d["shots"],d["shots_ag"],d["matches"],d["players"]
    GA=d["GA"]; ins=[]

    if not shots.empty:
        xgt=shots["xg"].sum(); gg=(shots["shot_type"]=="goal").sum()
        diff=gg-xgt
        if diff<-3: ins.append(("⚠️","Bitiricilik Krizi",
            f"xG {xgt:.1f} iken {int(gg)} gol — {diff:.1f} kayıp fırsat."))
        elif diff>3: ins.append(("✅","Bitiricilik Üstünlüğü",
            f"xG {xgt:.1f} iken {int(gg)} gol — beklentinin üzerinde."))
        far=(shots["player_coord_x"]>=25).sum()/max(len(shots),1)
        if far>0.25: ins.append(("⚠️","Uzak Mesafe Şut Fazlalığı",
            f"Şutların %{far*100:.0f}'i 25m+ uzaktan."))
        dt=shots[shots["situation"].isin(["set-piece","throw-in-set-piece","corner"])]
        if len(dt)/max(len(shots),1)>0.22: ins.append(("✅","Duran Top Etkinliği",
            f"Şutların %{len(dt)/len(shots)*100:.0f}'i duran toptan."))

    ev=matches[matches["side"]=="home"]; dep=matches[matches["side"]=="away"]
    ep=ev["res"].map({"G":3,"B":1,"M":0}).sum()
    dp=dep["res"].map({"G":3,"B":1,"M":0}).sum()
    if ep-dp>8: ins.append(("⚠️","Deplasman Zafiyeti",
        f"Evde {ep} puan, deplasmanda yalnızca {dp}."))
    if GA/max(len(matches),1)>1.1: ins.append(("⚠️","Savunma Açıkları",
        f"Maç başı {GA/len(matches):.2f} gol yeniliyor."))
    if not shots_ag.empty and len(shots_ag)/max(len(matches),1)>12:
        ins.append(("⚠️","Yüksek Şut Baskısı",
            f"Maç başı {len(shots_ag)/len(matches):.1f} rakip şutu."))

    if not players.empty:
        b=players[players["mac"]>=8].dropna(subset=["rat"]).nlargest(1,"rat")
        if not b.empty:
            r=b.iloc[0]
            ins.append(("⭐","En İyi Oyuncu",
                f"{r['player_name']} ({r['position']}) — {r['mac']} maçta {r['rat']:.2f} rating."))
        t=players.nlargest(1,"gol")
        if not t.empty and t.iloc[0]["gol"]>0:
            r=t.iloc[0]
            ins.append(("⚽","Top Golcü",
                f"{r['player_name']} — {int(r['gol'])} gol / {r['xg']:.1f} xG."))

    s5=matches.tail(5)["res"].tolist(); s5p=s5.count("G")*3+s5.count("B")
    if s5p<=4: ins.append(("📉","Kötü Form",f"Son 5 maçta {s5p} puan ({''.join(s5)})."))
    elif s5p>=10: ins.append(("📈","İyi Form",f"Son 5 maçta {s5p} puan ({''.join(s5)})."))
    return ins

# ── Sayfa yardımcıları ───────────────────────────────────────────────────────
def _section_header(fig, y, text, color=BL):
    fig.text(0.04, y, text, fontsize=11, fontweight="bold", color=color,
             transform=fig.transFigure)
    fig.add_artist(plt.Line2D([0.04,0.96],[y-0.012,y-0.012],
                              transform=fig.transFigure,color=color,lw=0.8,alpha=0.4))

def _stat_box(fig, x, y, value, label, color=OR, w=0.13, h=0.10):
    fig.add_artist(plt.Rectangle((x,y),w,h,transform=fig.transFigure,
                                  facecolor=S2,edgecolor=color,lw=1,alpha=0.9))
    fig.text(x+w/2,y+h*0.62,str(value),ha="center",fontsize=15,
             fontweight="bold",color=color,transform=fig.transFigure)
    fig.text(x+w/2,y+h*0.18,label,ha="center",fontsize=7.5,
             color=MU,transform=fig.transFigure)

def _form_strip(matches,n=10):
    last=matches.tail(n)
    parts=[]
    for _,r in last.iterrows():
        c={"G":GR,"B":OR,"M":RE}[r["res"]]
        parts.append((r["res"],c,r.get("opp","")))
    return parts

# ═══════════════════════════════════════════════════════════════════════════
# SAYFA FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════

def _p_cover(pdf, d:dict):
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    seas=d["matches"]["season"].iloc[0] if not d["matches"].empty else ""
    lg=d["matches"]["league"].iloc[0] if not d["matches"].empty else ""

    # Arkaplan degrade şerit
    grad=np.linspace(0,1,256).reshape(1,-1)
    ax.imshow(grad,extent=[0,16,9.5,11],aspect="auto",cmap="Blues",alpha=0.08,zorder=0)

    fig.text(0.5,0.73,d["team_name"].upper(),ha="center",fontsize=52,
             fontweight="bold",color=TX,transform=fig.transFigure)
    fig.text(0.5,0.64,f"{seas}  ·  {lg}",ha="center",fontsize=16,color=BL,
             transform=fig.transFigure)
    fig.text(0.5,0.57,"Kapsamlı Performans Raporu",ha="center",fontsize=12,color=MU,
             transform=fig.transFigure)

    # Ayırıcı çizgi
    fig.add_artist(plt.Line2D([0.1,0.9],[0.54,0.54],transform=fig.transFigure,
                              color=BL,lw=1,alpha=0.4))

    # Stat kutular
    W,D,L=d["W"],d["D"],d["L"]
    stats=[
        (str(len(d["matches"])),"Maç",BL),
        (f"{W}G {D}B {L}M","W-D-L",GR if W>L else RE),
        (str(d["PTS"]),"Puan",OR),
        (f"{d['GF']}-{d['GA']}","AG-YG",YE),
        (str(len(d["shots"])),"Şut",PU),
        (f"{d['TXG']:.1f}","Toplam xG",OR),
    ]
    xs=np.linspace(0.08,0.80,6)
    for i,((v,lbl,c),x) in enumerate(zip(stats,xs)):
        _stat_box(fig,x,0.36,v,lbl,color=c)

    # Form şeridi
    strip=_form_strip(d["matches"],10)
    fig.text(0.5,0.30,"Son 10 Maç",ha="center",fontsize=9,color=MU,
             transform=fig.transFigure)
    sw=0.038; gap=0.004; total=len(strip)*(sw+gap)
    sx0=0.5-total/2
    for i,(res,c,opp) in enumerate(strip):
        x=sx0+i*(sw+gap)
        fig.add_artist(plt.Rectangle((x,0.22),sw,0.055,transform=fig.transFigure,
                                      facecolor=c,edgecolor="none",alpha=0.85))
        fig.text(x+sw/2,0.247+0.003,res,ha="center",va="center",fontsize=10,
                 fontweight="bold",color=BG,transform=fig.transFigure)

    fig.text(0.5,0.06,"Footscrapping Analytics · Maç Verisi",ha="center",
             fontsize=8,color=MU,transform=fig.transFigure)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_results(pdf, d:dict):
    matches=d["matches"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.97,"Maç Sonuçları & Form Analizi",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.35,
                top=0.91,bottom=0.07,left=0.07,right=0.97)

    # Skor barları
    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor(S1)
    ax1.set_title(f"Haftalık Skor  —  {d['team_name']} (koyu) vs Rakip (soluk)",
                  fontsize=9,color=TX,loc="left",pad=5)
    n=len(matches); xi=np.arange(n)
    kg=matches["kg"].fillna(0).values; og=matches["og"].fillna(0).values
    for i,(res,k,o) in enumerate(zip(matches["res"],kg,og)):
        c={"G":GR,"B":OR,"M":RE}[res]
        ax1.bar(i-0.22,k,0.4,color=c,alpha=0.9,zorder=3)
        ax1.bar(i+0.22,o,0.4,color=MU,alpha=0.4,zorder=3)
        ax1.text(i,max(k,o)+0.12,res,ha="center",fontsize=6.5,color=c,
                 fontweight="bold",zorder=4)
    xtl=matches["round"].fillna("?").astype(str).tolist()
    ax1.set_xticks(xi); ax1.set_xticklabels(xtl,fontsize=6.5,rotation=0)
    ax1.set_xlabel("Hafta",color=MU,fontsize=8); ax1.set_ylabel("Gol",color=MU,fontsize=8)
    ax1.grid(axis="y",alpha=0.25,zorder=0); _sp(ax1); _border(ax1)
    _cap(ax1,"Haftalık skor grafiği. Renkli bar = takım golü, soluk bar = rakip golü. Üstteki harf maç sonucunu (G/B/M) gösterir.")

    # Kümülatif puan
    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(S1)
    ax2.set_title("Kümülatif Puan",fontsize=9,color=TX,loc="left",pad=5)
    pts_s=matches["res"].map({"G":3,"B":1,"M":0}).cumsum()
    r2=range(1,n+1); ideal=np.array(r2)*1.5
    ax2.plot(r2,pts_s.values,color=BL,lw=2.5,zorder=4)
    ax2.plot(r2,ideal,color=BR,lw=1.5,ls="--",zorder=3)
    ax2.fill_between(r2,pts_s.values,ideal,where=pts_s.values>=ideal,alpha=0.12,color=GR,zorder=2)
    ax2.fill_between(r2,pts_s.values,ideal,where=pts_s.values<ideal,alpha=0.12,color=RE,zorder=2)
    ax2.legend(["Gerçek","1.5 pt/maç"],facecolor=S2,edgecolor=BR,fontsize=7)
    ax2.grid(alpha=0.25); _sp(ax2); _border(ax2)
    _cap(ax2,"Sezon boyunca biriken puan trendi. Kesikli hat = maç başı 1.5 pt ideali. Yeşil alan idealin üstünü, kırmızı alan altında kaldığı dönemleri gösterir.")

    # Ev/Dep
    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(S1)
    ax3.set_title("Ev / Deplasman",fontsize=9,color=TX,loc="left",pad=5)
    em=matches[matches["side"]=="home"]; dm=matches[matches["side"]=="away"]
    cats=["G","B","M","AG","YG"]
    ev_v=[(em["res"]=="G").sum(),(em["res"]=="B").sum(),(em["res"]=="M").sum(),
          em["kg"].sum(),em["og"].sum()]
    dp_v=[(dm["res"]=="G").sum(),(dm["res"]=="B").sum(),(dm["res"]=="M").sum(),
          dm["kg"].sum(),dm["og"].sum()]
    xi2=np.arange(5)
    ax3.bar(xi2-0.22,ev_v,0.42,color=BL,alpha=0.85,label=f"Ev ({len(em)})")
    ax3.bar(xi2+0.22,dp_v,0.42,color=PU,alpha=0.85,label=f"Dep ({len(dm)})")
    for i,(e,d2) in enumerate(zip(ev_v,dp_v)):
        ax3.text(i-0.22,e+0.15,str(int(e)),ha="center",fontsize=8,color=TX)
        ax3.text(i+0.22,d2+0.15,str(int(d2)),ha="center",fontsize=8,color=TX)
    ax3.set_xticks(xi2); ax3.set_xticklabels(cats,fontsize=9)
    ax3.legend(facecolor=S2,edgecolor=BR,fontsize=7)
    ax3.grid(axis="y",alpha=0.25); _sp(ax3); _border(ax3)
    _cap(ax3,"Ev ve deplasman maçlarındaki galibiyet/beraberlik/mağlubiyet ile gol sayılarının kırılımı. Ev avantajı ya da dezavantajını ortaya koyar.")

    # Son 10 rakip tablosu
    ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(S1); ax4.axis("off")
    ax4.set_title("Son Maçlar",fontsize=9,color=TX,loc="left",pad=5)
    last=matches.tail(10)[["opp","kg","og","res"]].copy()
    y_pos=0.93
    ax4.text(0.02,y_pos,"Rakip",transform=ax4.transAxes,fontsize=7.5,
             color=MU,fontweight="bold")
    ax4.text(0.72,y_pos,"Skor",transform=ax4.transAxes,fontsize=7.5,
             color=MU,fontweight="bold",ha="center")
    ax4.text(0.92,y_pos,"S",transform=ax4.transAxes,fontsize=7.5,
             color=MU,fontweight="bold",ha="center")
    y_pos-=0.06
    for _,row in last.iterrows():
        c={"G":GR,"B":OR,"M":RE}[row["res"]]
        opp=str(row["opp"])[:22]
        ax4.text(0.02,y_pos,opp,transform=ax4.transAxes,fontsize=7,color=TX)
        ax4.text(0.72,y_pos,f"{int(row['kg'])}-{int(row['og'])}",
                 transform=ax4.transAxes,fontsize=7,color=TX,ha="center")
        ax4.text(0.92,y_pos,row["res"],transform=ax4.transAxes,
                 fontsize=7.5,color=c,fontweight="bold",ha="center")
        y_pos-=0.085
    _sp(ax4); _border(ax4)
    _cap(ax4,"Son 10 maçın rakip, skor ve sonuç özeti. Güncel form ve rakip kalitesini hızlıca değerlendirmek için kullanılır.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_shotmaps(pdf, d:dict):
    # figsize dikey (11×20) → pitch oranı (58:106≈0.547) hücrelere tam oturuyor
    fig=plt.figure(figsize=(11,20)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.985,"Şut & Yoğunluk Haritaları",ha="center",
             fontsize=13,fontweight="bold",color=TX)

    gs=GridSpec(2,2,figure=fig,hspace=0.14,wspace=0.10,
                top=0.975,bottom=0.025,left=0.03,right=0.97)

    ax1=fig.add_subplot(gs[0,0])
    _shotmap(ax1,d["shots"],f"⚽  {d['team_name']} — Hücum Şutları"); _border(ax1)
    _cap(ax1,"Takımın attığı her şutun sahada konumu. Nokta boyutu xG değeriyle orantılıdır; ★ gol, daire kurtarış/kaçan şutu gösterir.")

    ax2=fig.add_subplot(gs[0,1])
    _shotmap(ax2,d["shots_ag"],"🛡️  Rakip Şutları (Savunma)"); _border(ax2)
    _cap(ax2,"Rakibin attığı şutların konumu. Takımın savunma bölgesindeki hangi alanlarda baskıya maruz kaldığını gösterir.")

    ax3=fig.add_subplot(gs[1,0])
    _heatmap(ax3,d["shots"],"🔥  Hücum Yoğunluğu",[OR,"#ff4500","#ff0000"]); _border(ax3)
    _cap(ax3,"Hücum şutlarının KDE yoğunluk haritası. Sıcak renkler en çok şut üretilen bölgeleri, % değerleri o bölgenin toplam içindeki payını gösterir.")

    ax4=fig.add_subplot(gs[1,1])
    _heatmap(ax4,d["shots_ag"],"🔵  Savunma Baskı Yoğunluğu",[BL,PU,"#8800ff"]); _border(ax4)
    _cap(ax4,"Rakip şutlarının yoğunluk dağılımı. Mor/mavi bölgeler savunmanın en fazla zorlandığı alanları ve potansiyel açıkları işaret eder.")

    fig.text(0.5,0.010,
             "Boyut = xG değeri  ·  ★=Gol  ·  % rakamları bölgeye düşen şut oranı",
             ha="center",fontsize=7.5,color=MU)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _shot_overlay(ax, df, title, colors):
    """KDE ısı haritasını arka plan, şut noktalarını üst katman olarak çizer."""
    _half_pitch(ax, bg=BG)
    ax.set_title(title, fontsize=9, color=TX, pad=5, loc="left")
    df = df.dropna(subset=["player_coord_x", "player_coord_y"])
    if len(df) >= 5:
        x = df["player_coord_x"].clip(0, 50).values
        y = df["player_coord_y"].clip(0, 100).values
        xi = np.linspace(0, 50, 80); yi = np.linspace(0, 100, 80)
        Xi, Yi = np.meshgrid(xi, yi)
        try:
            k = gaussian_kde(np.vstack([x, y]), bw_method=0.28)
            Z = k(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            cmap = LinearSegmentedColormap.from_list("hm", [BG] + colors)
            ax.contourf(Xi, Yi, Z, levels=18, cmap=cmap, alpha=0.60, zorder=3)
        except Exception:
            pass
    if df.empty:
        return
    type_cfg = {
        "goal":  dict(color=GR,  base_s=160, alpha=0.95, marker="*", edge="white", zorder=12),
        "save":  dict(color="white", base_s=35,  alpha=0.55, marker="o", edge="none",  zorder=8),
        "miss":  dict(color=OR,  base_s=25,  alpha=0.45, marker="o", edge="none",  zorder=7),
        "block": dict(color=PU,  base_s=25,  alpha=0.45, marker="s", edge="none",  zorder=7),
    }
    handles = []
    for t, g in df.groupby("shot_type"):
        cfg = type_cfg.get(t, dict(color=MU, base_s=20, alpha=0.35, marker="o", edge="none", zorder=6))
        sizes = (g["xg"].fillna(0.05).clip(0.01, 1) * cfg["base_s"] * 3).values if "xg" in g.columns else cfg["base_s"]
        ax.scatter(g["player_coord_x"], g["player_coord_y"],
                   c=cfg["color"], s=sizes, alpha=cfg["alpha"],
                   marker=cfg["marker"], edgecolors=cfg["edge"], lw=0.7, zorder=cfg["zorder"])
        handles.append(mpatches.Patch(color=cfg["color"], label=t, alpha=0.8))
    ax.legend(handles=handles, loc="upper right", facecolor=S2, edgecolor=BR, fontsize=6.5,
              framealpha=0.9, ncol=2)
    total_xg = df["xg"].sum() if "xg" in df.columns else 0
    goals_cnt = (df["shot_type"] == "goal").sum()
    ax.text(0.02, 0.97, f"xG {total_xg:.1f} | {int(goals_cnt)} gol | {len(df)} şut",
            transform=ax.transAxes, fontsize=7, color=MU, ha="left", va="top", zorder=15)


def _p_shot_overlay(pdf, d: dict):
    """Şut haritası + yoğunluk haritasının üst üste bindirilmiş hali."""
    fig = plt.figure(figsize=(11, 12)); fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.985, "Şut + Yoğunluk Bindirmeli Harita",
             ha="center", fontsize=13, fontweight="bold", color=TX)
    gs = GridSpec(1, 2, figure=fig, hspace=0.10, wspace=0.10,
                  top=0.965, bottom=0.03, left=0.03, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    _shot_overlay(ax1, d["shots"], f"⚽  {d['team_name']} — Hücum", [OR, "#ff4500", "#ff0000"])
    _border(ax1)
    _cap(ax1, "Şut konumları ve yoğunluk haritası üst üste. Isı → en çok şut üretilen bölge; ★ gol noktaları.")

    ax2 = fig.add_subplot(gs[0, 1])
    _shot_overlay(ax2, d["shots_ag"], "🛡️  Rakip — Savunma", [BL, PU, "#8800ff"])
    _border(ax2)
    _cap(ax2, "Rakibin şut yoğunluğu + nokta konumları. Kırmızı/mor alan savunmanın en çok zorlandığı bölge.")

    pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)


def _p_xg(pdf, d:dict):
    shots,shots_ag,xgd=d["shots"],d["shots_ag"],d["xgd"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Kale Ağzı & xG Analizi",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,
                top=0.92,bottom=0.07,left=0.06,right=0.97)

    def gm_plot(ax, df, title):
        _goal_frame(ax)
        ax.set_title(title, fontsize=9, color=TX, pad=4, loc="left")
        df = df.dropna(subset=["goal_mouth_y"])
        if df.empty: return
        # goal_mouth_x DB'de dolu gelmiyor; xgot yükseklik proxy'si olarak kullanılır
        # (yüksek xgot → köşeye/zor konuma atılan şut → daha yüksek görsel konumda)
        rng = np.random.default_rng(0)
        def xy(row):
            gx = 36.8 + (row["goal_mouth_y"] / 100) * (63.2 - 36.8)
            xgot_val = float(row.get("xgot") or 0.0)
            # xgot 0-1 → kale yüksekliği 0-2.44 + küçük deterministik titreşim
            gy = np.clip(xgot_val * 2.44, 0.05, 2.39) + rng.uniform(-0.18, 0.18)
            return gx, float(np.clip(gy, 0.02, 2.42))
        saves = df[df["shot_type"] == "save"]
        goals = df[df["shot_type"] == "goal"]
        if not saves.empty:
            xs, ys = zip(*[xy(r) for _, r in saves.iterrows()])
            ax.scatter(xs, ys, c=BL, s=55, alpha=0.65, edgecolors=BG, lw=0.5, zorder=8, label="Kurtarış")
        if not goals.empty:
            xg2, yg = zip(*[xy(r) for _, r in goals.iterrows()])
            ax.scatter(xg2, yg, c=GR, s=130, alpha=0.95, marker="*",
                       edgecolors="white", lw=0.7, zorder=10, label="Gol")
            for xi, yi, (_, row) in zip(xg2, yg, goals.iterrows()):
                if row.get("xg"):
                    ax.text(xi, yi + 0.20, f"{row['xg']:.2f}", ha="center",
                            fontsize=5.5, color=YE, zorder=11)
        ax.legend(loc="upper right", facecolor=S2, edgecolor=BR, fontsize=7,
                  framealpha=0.9)
        _border(ax)

    ax1=fig.add_subplot(gs[0,0]); gm_plot(ax1,shots,"⚽  Atılan Goller")
    _cap(ax1,"Atılan gollerin kale içindeki yerleşimi. ★ gol, daire kurtarışı gösterir. Yanındaki sarı sayı o pozisyonun xG değeridir (golün zorluk derecesi).")
    ax2=fig.add_subplot(gs[0,1]); gm_plot(ax2,shots_ag,"🧤  Yenilen Goller")
    _cap(ax2,"Yenilen gollerin kale içindeki konumu. Kalecinin hangi köşede ve yükseklikte en çok zorlandığını ortaya koyar.")

    # xG vs Gol maç bazlı
    ax3=fig.add_subplot(gs[0,2]); ax3.set_facecolor(S1)
    ax3.set_title("xG vs Gol (Maç Bazlı)",fontsize=9,color=TX,loc="left",pad=4)
    if not xgd.empty:
        r3=range(1,len(xgd)+1)
        ax3.bar([i-0.22 for i in r3],xgd["xg"],0.42,color=OR,alpha=0.8,label="xG")
        ax3.bar([i+0.22 for i in r3],xgd["gol"],0.42,color=GR,alpha=0.8,label="Gol")
        ax3_r=ax3.twinx()
        cumxg=xgd["xg"].cumsum(); cumg=xgd["gol"].cumsum()
        ax3_r.plot(r3,(cumg-cumxg).values,color=RE,lw=1.5,ls="--",alpha=0.7)
        ax3_r.axhline(0,color=BR,lw=0.8,ls=":",alpha=0.5)
        ax3_r.set_ylabel("Kümülatif G-xG",color=RE,fontsize=7)
        ax3_r.tick_params(axis="y",colors=RE,labelsize=7)
        ax3.legend(facecolor=S2,edgecolor=BR,fontsize=7)
        ax3.grid(axis="y",alpha=0.25)
    _sp(ax3); _border(ax3)
    _cap(ax3,"Maç bazında beklenen gol (turuncu) ile gerçek gol (yeşil) karşılaştırması. Kesikli kırmızı = kümülatif G−xG farkı; pozitif = bitiricilik üstünlüğü.")

    # Şut kalitesi
    ax4=fig.add_subplot(gs[1,0]); ax4.set_facecolor(S1)
    ax4.set_title("Şut Kalitesi (xG Bandı)",fontsize=9,color=TX,loc="left",pad=4)
    if not shots.empty:
        xgv=shots["xg"].dropna()
        bins=[0,0.05,0.1,0.2,0.35,0.5,1.0]
        lbls=["<5%","5-10%","10-20%","20-35%","35-50%",">50%"]
        cnts=[((xgv>=bins[i])&(xgv<bins[i+1])).sum() for i in range(len(bins)-1)]
        clrs=[MU,RE,OR,YE,GR,BL]
        bars=ax4.barh(lbls,cnts,color=clrs,alpha=0.85,edgecolor=BG,lw=0.5)
        for b,v in zip(bars,cnts):
            ax4.text(v+0.3,b.get_y()+b.get_height()/2,str(int(v)),
                     va="center",fontsize=8,color=TX)
        ax4.set_xlabel("Şut Sayısı",color=MU,fontsize=8)
        ax4.grid(axis="x",alpha=0.25)
    _sp(ax4); _border(ax4)
    _cap(ax4,"Şutların xG değer bandına göre dağılımı. Yeşil/mavi satırlar (%35+) yüksek kaliteli açık pozisyon demektir; kırmızı/gri düşük kaliteyi gösterir.")

    # Durum analizi
    ax5=fig.add_subplot(gs[1,1]); ax5.set_facecolor(S1)
    ax5.set_title("Oyun Durumu",fontsize=9,color=TX,loc="left",pad=4)
    if not shots.empty:
        nm={"regular":"Normal","assisted":"Asiste","set-piece":"Duran Top",
            "corner":"Korner","throw-in-set-piece":"Taç Duran","counter-attack":"Kontr."}
        sit=shots["situation"].map(nm).fillna(shots["situation"]).value_counts().head(6)
        ax5.barh(sit.index,sit.values,color=BL,alpha=0.82,edgecolor=BG,lw=0.5)
        for i,v in enumerate(sit.values):
            ax5.text(v+0.3,i,str(int(v)),va="center",fontsize=8,color=TX)
        ax5.grid(axis="x",alpha=0.25)
    _sp(ax5); _border(ax5)
    _cap(ax5,"Şutların kaynaklandığı oyun durumu. Duran top ve korner etkinliği ile normal oyun ve kontr-atak kaynaklı tehdit dağılımını gösterir.")

    # Vücut bölgesi
    ax6=fig.add_subplot(gs[1,2]); ax6.set_facecolor(S1)
    ax6.set_title("Vücut Bölgesi",fontsize=9,color=TX,loc="left",pad=4)
    if not shots.empty:
        nm2={"right-foot":"Sağ Ayak","left-foot":"Sol Ayak","head":"Kafa"}
        bp=shots["body_part"].map(nm2).fillna(shots["body_part"]).value_counts()
        ax6.bar(bp.index,bp.values,color=[GR,OR,BL,PU][:len(bp)],alpha=0.85,edgecolor=BG,lw=0.5)
        for b in ax6.patches:
            ax6.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,
                     str(int(b.get_height())),ha="center",fontsize=9,color=TX)
        ax6.grid(axis="y",alpha=0.25)
    _sp(ax6); _border(ax6)
    _cap(ax6,"Şutların hangi vücut bölgesiyle atıldığı. Sağ/sol ayak ağırlığı takımın hücum tercihini, kafa oranı yüksek top oyununa bağımlılığı yansıtır.")
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_players(pdf, d:dict):
    players=d["players"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Oyuncu Performans Analizi",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,
                top=0.92,bottom=0.07,left=0.08,right=0.97)

    p5=players[players["mac"]>=5].copy().fillna(0)

    # Rating barı
    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor(S1)
    ax1.set_title("Rating Sıralaması (min 5 maç)",fontsize=9,color=TX,loc="left",pad=5)
    pr=p5[p5["rat"]>0].nlargest(18,"rat")
    if not pr.empty:
        colors=[_pc(r) for r in pr["position"]]
        bars=ax1.barh(pr["player_name"],pr["rat"],color=colors,alpha=0.85,
                      edgecolor=BG,lw=0.4,height=0.65)
        for bar,v in zip(bars,pr["rat"]):
            ax1.text(v+0.01,bar.get_y()+bar.get_height()/2,f"{v:.2f}",
                     va="center",fontsize=7.5,color=TX)
        ax1.axvline(6.5,color=MU,ls="--",lw=1,alpha=0.6)
        ax1.set_xlim(5.5,max(8.0,pr["rat"].max()+0.2))
    lp=[mpatches.Patch(color=YE,label="K"),mpatches.Patch(color=BL,label="D"),
        mpatches.Patch(color=GR,label="OS"),mpatches.Patch(color=RE,label="F")]
    ax1.legend(handles=lp,facecolor=S2,edgecolor=BR,fontsize=8,loc="lower right",
               title="Mevki",title_fontsize=7,ncol=4)
    ax1.grid(axis="x",alpha=0.25); _sp(ax1); _border(ax1)
    _cap(ax1,"Maç bazlı performans puanının sezon ortalaması (min 5 maç). 6.5 kesikli hattı lig ortalamasını temsil eder. Renk oyuncunun mevkisini gösterir.")

    # xG vs Gol scatter
    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(S1)
    ax2.set_title("Gol vs xG (Bireysel)",fontsize=9,color=TX,loc="left",pad=4)
    forv=p5[p5["xg"]>0].copy()
    if not forv.empty:
        ax2.scatter(forv["xg"],forv["gol"],c=[_pc(r) for r in forv["position"]],
                    s=forv["mac"]*8,alpha=0.82,edgecolors="white",lw=0.4,zorder=5)
        mv=max(forv["xg"].max(),forv["gol"].max())+0.5
        ax2.plot([0,mv],[0,mv],color=BR,ls="--",lw=1)
        for _,row in forv[forv["xg"]>1.5].iterrows():
            ax2.annotate(row["player_name"].split()[-1],(row["xg"],row["gol"]),
                         fontsize=6.5,color=TX,xytext=(4,4),textcoords="offset points")
        ax2.set_xlabel("xG",color=MU,fontsize=8); ax2.set_ylabel("Gol",color=MU,fontsize=8)
        ax2.text(0.03,0.92,"↑ xG üstü",transform=ax2.transAxes,fontsize=7,color=GR)
        ax2.text(0.55,0.05,"↓ xG altı",transform=ax2.transAxes,fontsize=7,color=RE)
        ax2.grid(alpha=0.25)
    _sp(ax2); _border(ax2)
    _cap(ax2,"Bireysel oyuncuların beklenen gol (xG) ile gerçek gol karşılaştırması. Diyagonal üstü bitiricilik üstünlüğünü, altı ise kaçan fırsatları gösterir. Nokta boyutu oynanan maç sayısıyla orantılıdır.")

    # Gol katkısı p90
    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(S1)
    ax3.set_title("Gol Katkısı p90 (min 300dk)",fontsize=9,color=TX,loc="left",pad=4)
    c3=p5[p5["dk"]>=300].copy()
    if not c3.empty:
        c3["gkp90"]=(c3["gol"]+c3["ast"])/c3["dk"]*90
        c3=c3.nlargest(12,"gkp90")
        bars=ax3.barh(c3["player_name"],c3["gkp90"],
                      color=[_pc(r) for r in c3["position"]],alpha=0.85,edgecolor=BG,lw=0.4)
        for bar,v in zip(bars,c3["gkp90"]):
            ax3.text(v+0.003,bar.get_y()+bar.get_height()/2,f"{v:.2f}",
                     va="center",fontsize=7.5,color=TX)
        ax3.set_xlabel("(Gol+Asist)/90dk",color=MU,fontsize=8)
        ax3.grid(axis="x",alpha=0.25)
    _sp(ax3); _border(ax3)
    _cap(ax3,"90 dakikaya normalize edilmiş (Gol+Asist) katkısı. Oyun süresinden bağımsız olarak en etkili hücum oyuncularını karşılaştırır.")

    # Anahtar pas
    ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(S1)
    ax4.set_title("Anahtar Pas (Toplam)",fontsize=9,color=TX,loc="left",pad=4)
    c4=p5[p5["dk"]>=200].copy() if not p5.empty else p5
    if not c4.empty and "kpas" in c4.columns:
        c4=c4.nlargest(12,"kpas")
        bars=ax4.barh(c4["player_name"],c4["kpas"],
                      color=[_pc(r) for r in c4["position"]],alpha=0.85,edgecolor=BG,lw=0.4)
        for bar,v in zip(bars,c4["kpas"]):
            ax4.text(v+0.1,bar.get_y()+bar.get_height()/2,f"{v:.0f}",
                     va="center",fontsize=7.5,color=TX)
        ax4.grid(axis="x",alpha=0.25)
    _sp(ax4); _border(ax4)
    _cap(ax4,"Şuta dönüşen son pas (anahtar pas) sayısı toplamı. Oyun kurucuları ve yaratıcı kanatları belirler; asiste dönüşmese de fırsat üretimini ölçer.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_insights(pdf, d:dict, insights:list):
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Otomatik Performans Değerlendirmesi",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    W,D,L,PTS,GF,GA=d["W"],d["D"],d["L"],d["PTS"],d["GF"],d["GA"]
    summ=(f"  {len(d['matches'])} Maç  ·  {W}G {D}B {L}M  ·  {PTS} Puan  ·  "
          f"AG:{GF} YG:{GA}  ·  xG:{d['TXG']:.1f}  ·  Şut:{len(d['shots'])}")
    fig.text(0.5,0.915,summ,ha="center",fontsize=10,color=TX,
             bbox=dict(boxstyle="round,pad=0.5",facecolor=S2,edgecolor=BL,lw=1.5),
             transform=fig.transFigure)

    goods=[i for i in insights if i[0] in("✅","⭐","📈","⚽")]
    bads =[i for i in insights if i[0] in("⚠️","📉")]

    fig.add_artist(plt.Rectangle((0.03,0.10),0.455,0.76,transform=fig.transFigure,
                                  facecolor=S1,edgecolor=f"#1a4a2a",lw=1.5))
    fig.add_artist(plt.Rectangle((0.515,0.10),0.455,0.76,transform=fig.transFigure,
                                  facecolor=S1,edgecolor=f"#4a1a1a",lw=1.5))

    fig.text(0.06,0.835,"💪  GÜÇLÜ YÖNLER",fontsize=11,fontweight="bold",color=GR,
             transform=fig.transFigure)
    fig.text(0.545,0.835,"⚠️  GELİŞTİRİLECEK ALANLAR",fontsize=11,fontweight="bold",
             color=RE,transform=fig.transFigure)

    y=0.79
    for ico,title,desc in (goods or [("—","Veri yok","Yetersiz veri.")]):
        fig.text(0.06,y,f"{ico}  {title}",fontsize=9.5,fontweight="bold",color=GR,
                 transform=fig.transFigure); y-=0.042
        for chunk in [desc[i:i+72] for i in range(0,len(desc),72)]:
            fig.text(0.07,y,chunk,fontsize=8,color=TX,transform=fig.transFigure)
            y-=0.028
        y-=0.01

    y=0.79
    for ico,title,desc in (bads or [("—","Veri yok","Yetersiz veri.")]):
        fig.text(0.545,y,f"{ico}  {title}",fontsize=9.5,fontweight="bold",color=OR,
                 transform=fig.transFigure); y-=0.042
        for chunk in [desc[i:i+72] for i in range(0,len(desc),72)]:
            fig.text(0.555,y,chunk,fontsize=8,color=TX,transform=fig.transFigure)
            y-=0.028
        y-=0.01

    fig.text(0.5,0.04,"Footscrapping Analytics",
             ha="center",fontsize=8,color=MU,transform=fig.transFigure)
    info=pdf.infodict(); info["Title"]=f"{d['team_name']} Raporu"; info["Author"]="Footscrapping"
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# KARŞILAŞTIRMA SAYFALARI
# ═══════════════════════════════════════════════════════════════════════════

def _ts_avg(d:dict, col:str)->float:
    ts=d["team_st"]
    if ts.empty or col not in ts.columns: return 0.0
    return float(ts[col].mean())

def _winning_tips(da:dict, db2:dict)->list:
    ta,tb=da["team_name"],db2["team_name"]; tips=[]
    a_xg=da["TXG"]/max(len(da["matches"]),1)
    b_xga=float(db2["shots_ag"]["xg"].sum())/max(len(db2["matches"]),1) if not db2["shots_ag"].empty else 0
    if a_xg>1.2 and b_xga>1.0:
        tips.append(f"✅ {ta} yüksek xG üretiyor ({a_xg:.2f}/maç).\n   {tb}'ın savunması baskıya açık ({b_xga:.2f} xGA/maç). Hızlı geçişlerle gol arayın.")
    a_pos=_ts_avg(da,"ball_possession"); b_pos=_ts_avg(db2,"ball_possession")
    if a_pos>b_pos+6:
        tips.append(f"✅ Top hakimiyetinde {ta} önde ({a_pos:.0f}% vs {b_pos:.0f}%).\n   Tempoyu kontrol edin, rakibi yüksek blokla sıkıştırın.")
    if not da["shots"].empty:
        sp=da["shots"]["situation"].isin(["set-piece","corner","throw-in-set-piece"]).mean()
        if sp>0.20:
            tips.append(f"✅ {ta} duran toptan güçlü (%{sp*100:.0f}).\n   Korner ve FK setlerine özel hazırlık yapın.")
    if not db2["shots_ag"].empty:
        b_box=(db2["shots_ag"]["player_coord_x"]<16.5).mean()
        if b_box>0.35:
            tips.append(f"⚠️ {tb}'a karşı ceza sahası içinden şutlarda dikkatli olun.\n   Rakip bu bölgeden %{b_box*100:.0f} oranında şut izin veriyor.")
    dep_b=db2["matches"][db2["matches"]["side"]=="away"]
    if not dep_b.empty:
        dp=dep_b["res"].map({"G":3,"B":1,"M":0}).mean()
        if dp<1.0:
            tips.append(f"✅ {tb} deplasmanda zayıf (ort. {dp:.1f} puan/maç).\n   Konumunuza göre erken gol baskısı yapın.")
    if not tips:
        tips.append("Yeterli istatistiksel fark bulunamadı.\nDaha fazla maç verisi analizi derinleştirecek.")
    return tips


def _p_h2h_cover(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    # Başlık
    fig.text(0.5,0.91,"KARŞILAŞTIRMA RAPORU",ha="center",fontsize=14,
             color=MU,fontweight="bold",transform=fig.transFigure)
    fig.text(0.28,0.82,ta.upper(),ha="center",fontsize=30,fontweight="bold",
             color=BL,transform=fig.transFigure)
    fig.text(0.5,0.82,"VS",ha="center",fontsize=22,color=MU,
             fontweight="bold",transform=fig.transFigure)
    fig.text(0.72,0.82,tb.upper(),ha="center",fontsize=30,fontweight="bold",
             color=RE,transform=fig.transFigure)

    fig.add_artist(plt.Line2D([0.1,0.9],[0.79,0.79],transform=fig.transFigure,
                              color=BR,lw=0.8))

    # İki taraf yan yana stat grid
    def side_stats(d,xbase,color):
        W,D,L=d["W"],d["D"],d["L"]
        stats=[
            (str(len(d["matches"])),"Maç"),
            (f"{W}G {D}B {L}M","Seri"),
            (str(d["PTS"]),"Puan"),
            (f"{d['GF']}-{d['GA']}","AG-YG"),
            (f"{d['TXG']:.1f}","xG"),
            (str(len(d["shots"])),"Şut"),
        ]
        for i,(v,lbl) in enumerate(stats):
            row,col=divmod(i,3)
            x=xbase+col*0.085; y=0.59-row*0.13
            fig.add_artist(plt.Rectangle((x,y),0.078,0.11,
                                          transform=fig.transFigure,
                                          facecolor=S2,edgecolor=color,lw=1,alpha=0.9))
            fig.text(x+0.039,y+0.072,v,ha="center",fontsize=13,fontweight="bold",
                     color=color,transform=fig.transFigure)
            fig.text(x+0.039,y+0.018,lbl,ha="center",fontsize=7,
                     color=MU,transform=fig.transFigure)

    side_stats(da,0.085,BL)
    side_stats(db2,0.555,RE)

    # Form şeritleri
    for d,xbase,color in [(da,0.085,BL),(db2,0.555,RE)]:
        strip=_form_strip(d["matches"],8)
        sw=0.036; gap=0.004
        for i,(res,c,_) in enumerate(strip):
            x=xbase+i*(sw+gap)
            fig.add_artist(plt.Rectangle((x,0.30),sw,0.048,
                                          transform=fig.transFigure,
                                          facecolor=c,edgecolor="none",alpha=0.85))
            fig.text(x+sw/2,0.323+0.001,res,ha="center",va="center",
                     fontsize=9,fontweight="bold",color=BG,transform=fig.transFigure)

    fig.text(0.085,0.36,"Son 8 Maç",fontsize=8,color=MU,transform=fig.transFigure)
    fig.text(0.555,0.36,"Son 8 Maç",fontsize=8,color=MU,transform=fig.transFigure)

    fig.text(0.5,0.06,"Footscrapping Analytics · Maç Verisi",ha="center",
             fontsize=8,color=MU,transform=fig.transFigure)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_metrics(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,f"Metrik Karşılaştırması: {ta}  vs  {tb}",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(1,2,figure=fig,wspace=0.04,top=0.91,bottom=0.07,left=0.03,right=0.97)

    metrics=[
        ("xG / Maç",         da["TXG"]/max(len(da["matches"]),1), db2["TXG"]/max(len(db2["matches"]),1), True),
        ("İsabetli Şut/Maç", _ts_avg(da,"shots_on_target"),       _ts_avg(db2,"shots_on_target"),       True),
        ("Topp. %",          _ts_avg(da,"ball_possession"),        _ts_avg(db2,"ball_possession"),       True),
        ("Pas İsabeti %",    _ts_avg(da,"accurate_passes_pct"),    _ts_avg(db2,"accurate_passes_pct"),   True),
        ("Anahtar Pas/Maç",  _ts_avg(da,"key_passes"),            _ts_avg(db2,"key_passes"),            True),
        ("Korner/Maç",       _ts_avg(da,"corner_kicks"),          _ts_avg(db2,"corner_kicks"),          True),
        ("Müdahale/Maç",     _ts_avg(da,"total_tackles"),         _ts_avg(db2,"total_tackles"),         True),
        ("Top Kesme/Maç",    _ts_avg(da,"interceptions"),         _ts_avg(db2,"interceptions"),         True),
        ("xGA / Maç",
         float(da["shots_ag"]["xg"].sum())/max(len(da["matches"]),1) if not da["shots_ag"].empty else 0,
         float(db2["shots_ag"]["xg"].sum())/max(len(db2["matches"]),1) if not db2["shots_ag"].empty else 0,
         False),  # düşük = iyi
        ("Yenilen Gol/Maç",  da["GA"]/max(len(da["matches"]),1), db2["GA"]/max(len(db2["matches"]),1), False),
        ("Faul/Maç",         _ts_avg(da,"fouls"),                 _ts_avg(db2,"fouls"),                 False),
    ]

    ax=fig.add_subplot(gs[0,:]); ax.set_facecolor(S1); ax.axis("off")

    # Başlık satırı
    ax.text(0.02,0.97,ta,transform=ax.transAxes,fontsize=10,
            fontweight="bold",color=BL,ha="left",va="top")
    ax.text(0.5,0.97,"Metrik",transform=ax.transAxes,fontsize=9,
            color=MU,ha="center",va="top")
    ax.text(0.98,0.97,tb,transform=ax.transAxes,fontsize=10,
            fontweight="bold",color=RE,ha="right",va="top")
    ax.plot([0.01,0.99],[0.94,0.94],color=BR,lw=0.8,transform=ax.transAxes,
            solid_capstyle="butt")

    row_h=0.076
    for i,(lbl,va2,vb2,higher_better) in enumerate(metrics):
        y=0.90-i*row_h
        total=max(va2+vb2,0.001)
        frac_a=va2/total if total>0 else 0.5
        frac_b=1-frac_a

        # A barı (sol → merkeze)
        bar_w=0.34
        ax.barh(y,frac_a*bar_w,left=0.5-frac_a*bar_w,height=row_h*0.55,
                color=BL if (higher_better and va2>=vb2) or (not higher_better and va2<=vb2) else "#2a3a4a",
                alpha=0.85,align="center",transform=ax.transAxes)
        # B barı (merkeze → sağa)
        ax.barh(y,frac_b*bar_w,left=0.5,height=row_h*0.55,
                color=RE if (higher_better and vb2>va2) or (not higher_better and vb2<va2) else "#4a2a2a",
                alpha=0.85,align="center",transform=ax.transAxes)

        # Değerler
        fmt=lambda v: f"{v:.2f}" if v<10 else f"{v:.1f}"
        ax.text(0.48,y,fmt(va2),transform=ax.transAxes,fontsize=8.5,
                color=TX,ha="right",va="center",fontweight="bold")
        ax.text(0.52,y,fmt(vb2),transform=ax.transAxes,fontsize=8.5,
                color=TX,ha="left",va="center",fontweight="bold")
        ax.text(0.5,y-row_h*0.28,lbl,transform=ax.transAxes,fontsize=7,
                color=MU,ha="center",va="center")

        # Ayırıcı
        ax.plot([0.01,0.99],[y-row_h*0.47,y-row_h*0.47],color=BR,lw=0.4,
                alpha=0.5,transform=ax.transAxes)

    _sp(ax); _border(ax)
    _cap(ax,"İki takımın 11 temel istatistik metriğindeki maç başına ortalama değerleri. Renkli çubuk üstün tarafı gösterir; değerler her satırın ortasında karşılıklı yazılmıştır.")
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_shots(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    # figsize dikey (11×20) → pitch oranı (58:106≈0.547) hücrelere tam oturuyor
    fig=plt.figure(figsize=(11,20)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.985,"Şut Analizi — Hücum & Savunma",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(2,2,figure=fig,hspace=0.10,wspace=0.10,
                top=0.975,bottom=0.02,left=0.03,right=0.97)

    ax1=fig.add_subplot(gs[0,0])
    _shotmap(ax1,da["shots"],f"⚽ {ta} — Hücum Şutları"); _border(ax1)
    _cap(ax1,"Atılan her şutun sahada konumu. Boyut xG değeriyle orantılıdır; ★ gol, daire kurtarış/kaçan şutu gösterir.")
    ax2=fig.add_subplot(gs[0,1])
    _heatmap(ax2,da["shots"],f"🔥 {ta} — Hücum Yoğunluğu",[OR,"#ff4500","#ff0000"]); _border(ax2)
    _cap(ax2,"Hücum şutlarının KDE yoğunluk haritası. % değerleri o bölgeden atılan şutların toplam içindeki oranını gösterir.")
    ax3=fig.add_subplot(gs[1,0])
    _shotmap(ax3,db2["shots"],f"⚽ {tb} — Hücum Şutları"); _border(ax3)
    _cap(ax3,"Atılan her şutun sahada konumu. Boyut xG değeriyle orantılıdır; ★ gol, daire kurtarış/kaçan şutu gösterir.")
    ax4=fig.add_subplot(gs[1,1])
    _heatmap(ax4,db2["shots"],f"🔥 {tb} — Hücum Yoğunluğu",[RE,OR,YE]); _border(ax4)
    _cap(ax4,"Hücum şutlarının KDE yoğunluk haritası. % değerleri o bölgeden atılan şutların toplam içindeki oranını gösterir.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_strategy(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    tips=_winning_tips(da,db2)
    ins_a=_insights(da); ins_b=_insights(db2)

    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,f"Taktik Analiz & Kazanma Stratejisi",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.55,wspace=0.35,
                top=0.92,bottom=0.07,left=0.04,right=0.97)

    # Radar karşılaştırma
    ax_r=fig.add_subplot(gs[0,:2],polar=True); ax_r.set_facecolor(S2)
    labels=["xG/Maç","İsab. Şut","Topposs%","Pas%","Korner","Müdahale"]
    def norm_vals(da2,db2_2):
        raw_a=[da2["TXG"]/max(len(da2["matches"]),1),_ts_avg(da2,"shots_on_target"),
               _ts_avg(da2,"ball_possession"),_ts_avg(da2,"accurate_passes_pct"),
               _ts_avg(da2,"corner_kicks"),_ts_avg(da2,"total_tackles")]
        raw_b=[db2_2["TXG"]/max(len(db2_2["matches"]),1),_ts_avg(db2_2,"shots_on_target"),
               _ts_avg(db2_2,"ball_possession"),_ts_avg(db2_2,"accurate_passes_pct"),
               _ts_avg(db2_2,"corner_kicks"),_ts_avg(db2_2,"total_tackles")]
        mx=[max(a,b,0.001) for a,b in zip(raw_a,raw_b)]
        return [v/m for v,m in zip(raw_a,mx)],[v/m for v,m in zip(raw_b,mx)]
    na,nb=norm_vals(da,db2)
    N=len(labels); angles=[n/N*2*np.pi for n in range(N)]; angles+=angles[:1]
    na+=na[:1]; nb+=nb[:1]
    ax_r.plot(angles,na,color=BL,lw=2.2); ax_r.fill(angles,na,color=BL,alpha=0.22)
    ax_r.plot(angles,nb,color=RE,lw=2.2); ax_r.fill(angles,nb,color=RE,alpha=0.22)
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(labels,color=TX,fontsize=8)
    ax_r.set_yticklabels([]); ax_r.spines["polar"].set_color(BR)
    ax_r.set_facecolor(S2)
    ax_r.legend([ta,tb],loc="upper right",bbox_to_anchor=(1.35,1.15),
                facecolor=S2,edgecolor=BR,fontsize=9,
                handles=[mpatches.Patch(color=BL,label=ta),
                         mpatches.Patch(color=RE,label=tb)])
    ax_r.set_title("Radar Karşılaştırması",fontsize=9,color=TX,pad=15)
    # (radar için _cap yerine fig.text — polar axes transAxes'i desteklemez)
    fig.text(0.285,0.505,"6 temel metrikte iki takımın normalize edilmiş değerleri. Dış çember = o grupta en iyi, iç çember = zayıf.",
             ha="center",fontsize=5.8,color=MU,style="italic")

    # Kazanma stratejisi metni
    ax_w=fig.add_subplot(gs[0,2]); ax_w.set_facecolor(S1); ax_w.axis("off")
    ax_w.set_title(f"{ta} İçin Kazanma Stratejisi",fontsize=9,color=GR,loc="left",pad=4)
    _sp(ax_w); _border(ax_w)
    y=0.92
    for tip in tips:
        for line in tip.split("\n"):
            ax_w.text(0.03,y,line,transform=ax_w.transAxes,fontsize=7.5,
                      color=TX,va="top"); y-=0.092
        y-=0.04
    _cap(ax_w,"İstatistiksel üstünlüklere ve rakip zafiyetlerine dayalı otomatik taktik öneriler. Modele girdi olarak xG, toppos, duran top ve deplasman verisi kullanılır.")

    # Güçlü yönler (A)
    ax_ga=fig.add_subplot(gs[1,0]); ax_ga.set_facecolor(S1); ax_ga.axis("off")
    ax_ga.set_title(f"💪 {ta} Güçlü Yönleri",fontsize=9,color=BL,loc="left",pad=4)
    _sp(ax_ga); _border(ax_ga)
    y=0.88
    for ico,title,desc in [i for i in ins_a if i[0] in("✅","⭐","📈","⚽")][:4]:
        ax_ga.text(0.03,y,f"{ico} {title}",transform=ax_ga.transAxes,
                   fontsize=8,color=GR,fontweight="bold"); y-=0.12
        ax_ga.text(0.05,y,desc[:60],transform=ax_ga.transAxes,
                   fontsize=7,color=TX); y-=0.12
    _cap(ax_ga,"Birinci takımın lig ortalamasının üzerinde performans gösterdiği, saha içinde avantaj sağlayan istatistiksel güçlü alanlar.")

    # Zayıf yönler (B)
    ax_wb=fig.add_subplot(gs[1,1]); ax_wb.set_facecolor(S1); ax_wb.axis("off")
    ax_wb.set_title(f"⚠️ {tb} Zayıf Yönleri",fontsize=9,color=RE,loc="left",pad=4)
    _sp(ax_wb); _border(ax_wb)
    y=0.88
    for ico,title,desc in [i for i in ins_b if i[0] in("⚠️","📉")][:4]:
        ax_wb.text(0.03,y,f"{ico} {title}",transform=ax_wb.transAxes,
                   fontsize=8,color=OR,fontweight="bold"); y-=0.12
        ax_wb.text(0.05,y,desc[:60],transform=ax_wb.transAxes,
                   fontsize=7,color=TX); y-=0.12
    _cap(ax_wb,"Rakip takımın geliştirmesi gereken ve rakip tarafından istismar edilebilecek istatistiksel zayıf noktalar.")

    # xG Yarışı
    ax_xg=fig.add_subplot(gs[1,2]); ax_xg.set_facecolor(S1)
    ax_xg.set_title("Kümülatif xG Karşılaştırması",fontsize=9,color=TX,loc="left",pad=4)
    for d,color,label in [(da,BL,ta),(db2,RE,tb)]:
        if not d["xgd"].empty:
            cumxg=d["xgd"]["xg"].cumsum()
            ax_xg.plot(range(1,len(cumxg)+1),cumxg.values,
                       color=color,lw=2,label=label)
    ax_xg.legend(facecolor=S2,edgecolor=BR,fontsize=7)
    ax_xg.set_xlabel("Maç",color=MU,fontsize=8)
    ax_xg.set_ylabel("Kümülatif xG",color=MU,fontsize=8)
    ax_xg.grid(alpha=0.25); _sp(ax_xg); _border(ax_xg)
    _cap(ax_xg,"Sezon boyunca biriken toplam beklenen gol (xG). Üstteki takım daha fazla kaliteli pozisyon üretmiş demektir; eğim maç başına üretim hızını yansıtır.")

    info=pdf.infodict()
    info["Title"]=f"{ta} vs {tb} Karşılaştırma"; info["Author"]="Footscrapping"
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

# ── "Neden Başarılı?" sayfası ──────────────────────────────────────────────
def _p_why_successful(pdf, d: dict, team_stats_df):
    """Stil parmak izi radar + güçlü/zayıf yön tablosu sayfası."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)

    # Başlık
    fig.text(0.5, 0.96, f"Neden Başarılı? — {d['team_name']}", ha="center",
             fontsize=15, color=BL, fontweight="bold")

    if team_stats_df.empty:
        fig.text(0.5, 0.5, "Takım istatistikleri yok.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    try:
        fp = _sfp(team_stats_df, d["team_name"])
    except Exception:
        fig.text(0.5, 0.5, "Stil parmak izi hesaplanamadı.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    if fp.empty:
        fig.text(0.5, 0.5, "Takım bulunamadı.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    sw = team_strengths_weaknesses(fp)

    # Sol: radar
    ax_rad = fig.add_axes([0.03, 0.08, 0.44, 0.82], polar=True)
    labels = list(fp.index)
    vals = [float(v) for v in fp.values]
    N = len(labels)
    if N > 2:
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        vals_plot = vals + vals[:1]
        ax_rad.set_facecolor(S1)
        ax_rad.set_theta_offset(np.pi / 2)
        ax_rad.set_theta_direction(-1)
        ax_rad.set_ylim(0, 100)
        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(labels, size=6.5, color=TX)
        ax_rad.set_yticks([25, 50, 75])
        ax_rad.set_yticklabels(["25", "50", "75"], size=5.5, color=MU)
        ax_rad.plot(angles, vals_plot, color=BL, linewidth=2)
        ax_rad.fill(angles, vals_plot, color=BL, alpha=0.25)
        ax_rad.set_title("Lig Percentile (%)", size=8, color=MU, pad=10)

    # Sağ: güçlü/zayıf tablo
    ax_tbl = fig.add_axes([0.50, 0.08, 0.48, 0.82])
    ax_tbl.set_facecolor(S1); ax_tbl.axis("off")
    _border(ax_tbl)

    ax_tbl.text(0.5, 0.97, "Güçlü Yönler", ha="center", va="top",
                fontsize=10, color=GR, fontweight="bold", transform=ax_tbl.transAxes)
    strengths = sw.get("strengths", [])
    for i, s in enumerate(strengths[:6]):
        y = 0.90 - i * 0.07
        ax_tbl.text(0.05, y, f"✅ {s['label']} — {s['value']:.0f}. percentile",
                    va="top", fontsize=8.5, color=GR, transform=ax_tbl.transAxes)

    ax_tbl.text(0.5, 0.90 - len(strengths[:6]) * 0.07 - 0.04,
                "Zayıf Yönler", ha="center", va="top",
                fontsize=10, color=RE, fontweight="bold", transform=ax_tbl.transAxes)
    weaknesses = sw.get("weaknesses", [])
    base_y = 0.90 - len(strengths[:6]) * 0.07 - 0.10
    for i, w in enumerate(weaknesses[:6]):
        y = base_y - i * 0.07
        ax_tbl.text(0.05, y, f"⚠️ {w['label']} — {w['value']:.0f}. percentile",
                    va="top", fontsize=8.5, color=OR, transform=ax_tbl.transAxes)

    if sw.get("summary"):
        ax_tbl.text(0.05, 0.03, sw["summary"], va="bottom", fontsize=7.5,
                    color=MU, transform=ax_tbl.transAxes, wrap=True)

    pdf.savefig(fig, dpi=120); plt.close(fig)


# ── H2H taktik önerileri sayfası ───────────────────────────────────────────
def _p_h2h_tactical(pdf, d_ours: dict, d_theirs: dict, team_stats_df):
    """İki takım için stil overlay radar + otomatik taktiksel öneri metinleri."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.96, f"H2H — {d_ours['team_name']} vs {d_theirs['team_name']}",
             ha="center", fontsize=14, color=BL, fontweight="bold")

    if team_stats_df.empty:
        fig.text(0.5, 0.5, "Takım istatistikleri yok.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    try:
        fp_a = _sfp(team_stats_df, d_ours["team_name"])
        fp_b = _sfp(team_stats_df, d_theirs["team_name"])
    except Exception:
        fp_a, fp_b = None, None

    # Sol: overlay radar
    ax_rad = fig.add_axes([0.03, 0.08, 0.44, 0.82], polar=True)
    ax_rad.set_facecolor(S1)
    if fp_a is not None and not fp_a.empty and fp_b is not None and not fp_b.empty:
        labels = list(fp_a.index)
        N = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax_rad.set_theta_offset(np.pi / 2); ax_rad.set_theta_direction(-1)
        ax_rad.set_ylim(0, 100)
        ax_rad.set_xticks(angles[:-1]); ax_rad.set_xticklabels(labels, size=6, color=TX)
        ax_rad.set_yticks([25, 50, 75]); ax_rad.set_yticklabels(["25","50","75"], size=5, color=MU)

        vals_a = list(fp_a.values) + [fp_a.values[0]]
        vals_b = list(fp_b.values) + [fp_b.values[0]]
        ax_rad.plot(angles, vals_a, color=BL, lw=2, label=d_ours["team_name"])
        ax_rad.fill(angles, vals_a, color=BL, alpha=0.2)
        ax_rad.plot(angles, vals_b, color=RE, lw=2, label=d_theirs["team_name"])
        ax_rad.fill(angles, vals_b, color=RE, alpha=0.2)
        ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                      fontsize=7, framealpha=0.3)

    # Sağ: taktik öneri metinleri
    ax_txt = fig.add_axes([0.50, 0.08, 0.48, 0.82])
    ax_txt.set_facecolor(S1); ax_txt.axis("off"); _border(ax_txt)

    if fp_a is not None and not fp_a.empty and fp_b is not None and not fp_b.empty:
        reco = h2h_tactical_recommendations(fp_a, fp_b,
                                            our_name=d_ours["team_name"],
                                            their_name=d_theirs["team_name"])
        exploit = [r for r in reco if r.get("type") == "exploit"]
        beware  = [r for r in reco if r.get("type") == "beware"]

        ax_txt.text(0.5, 0.97, "Nasıl Kazanılır?", ha="center", va="top",
                    fontsize=10, color=GR, fontweight="bold", transform=ax_txt.transAxes)
        for i, r in enumerate(exploit[:5]):
            y = 0.90 - i * 0.08
            ax_txt.text(0.04, y, f"✅ {r.get('text', '')}", va="top", fontsize=8,
                        color=GR, transform=ax_txt.transAxes, wrap=True)

        base = 0.90 - len(exploit[:5]) * 0.08 - 0.05
        ax_txt.text(0.5, base, "Dikkat Edilmesi Gerekenler", ha="center", va="top",
                    fontsize=10, color=OR, fontweight="bold", transform=ax_txt.transAxes)
        for i, r in enumerate(beware[:5]):
            y = base - 0.06 - i * 0.08
            ax_txt.text(0.04, y, f"⚠️ {r.get('text', '')}", va="top", fontsize=8,
                        color=OR, transform=ax_txt.transAxes, wrap=True)
    else:
        ax_txt.text(0.5, 0.5, "Stil verileri hesaplanamadı.", ha="center",
                    color=MU, transform=ax_txt.transAxes)

    pdf.savefig(fig, dpi=120); plt.close(fig)


# ── Oyuncu detay sayfası ───────────────────────────────────────────────────
def _p_player_detail(pdf, player_row: dict, all_players_df):
    """Tek oyuncu için radar + metrik tablo + benzer oyuncular."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)

    name = player_row.get("player_name", "?")
    pos  = player_row.get("position", "")
    mins = int(player_row.get("dk", 0) or 0)
    rat  = player_row.get("rat", 0) or 0
    mac  = int(player_row.get("mac", 0) or 0)

    fig.text(0.5, 0.96, name, ha="center", fontsize=14,
             color=_pc(str(pos)[:1].upper()), fontweight="bold")
    fig.text(0.5, 0.925, f"{pos} | {mac} maç | {mins} dk | Ort. {rat:.2f}",
             ha="center", fontsize=9, color=MU)

    # Radar: p90 metrikler vs lig ortalaması
    metrics_raw = {
        "Gol":     float(player_row.get("gol") or 0),
        "Asist":   float(player_row.get("ast") or 0),
        "xG":      float(player_row.get("xg") or 0),
        "xA":      float(player_row.get("xa") or 0),
        "Şut":     float(player_row.get("sut") or 0),
        "Pas":     float(player_row.get("pas") or 0),
        "Pres":    float(player_row.get("intp") or 0),
        "İkili":   float(player_row.get("dw") or 0),
        "İlerleme":float(player_row.get("prun") or 0),
    }
    # Per-90
    p90 = {k: v / mins * 90 if mins > 0 else 0 for k, v in metrics_raw.items()}

    if not all_players_df.empty:
        lig_avgs = {}
        col_map = {"Gol":"gol","Asist":"ast","xG":"xg","xA":"xa","Şut":"sut",
                   "Pas":"pas","Pres":"intp","İkili":"dw","İlerleme":"prun"}
        for label, col in col_map.items():
            if col in all_players_df.columns:
                dk_col = all_players_df["dk"].replace(0, np.nan)
                per90 = all_players_df[col].astype(float).fillna(0) / dk_col * 90
                lig_avgs[label] = float(per90.median())
            else:
                lig_avgs[label] = 0.0
        # Percentile rank
        pct = {}
        for label, col in col_map.items():
            if col in all_players_df.columns:
                dk_col = all_players_df["dk"].replace(0, np.nan)
                per90_series = all_players_df[col].astype(float).fillna(0) / dk_col * 90
                rank = (per90_series < p90[label]).sum() / max(per90_series.notna().sum(), 1) * 100
                pct[label] = round(rank, 1)
            else:
                pct[label] = 50.0
    else:
        pct = {k: 50.0 for k in p90}
        lig_avgs = {k: 0.0 for k in p90}

    ax_rad = fig.add_axes([0.03, 0.08, 0.40, 0.80], polar=True)
    ax_rad.set_facecolor(S1)
    labels = list(pct.keys()); N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_pct = list(pct.values()) + [list(pct.values())[0]]
    ax_rad.set_theta_offset(np.pi / 2); ax_rad.set_theta_direction(-1)
    ax_rad.set_ylim(0, 100)
    ax_rad.set_xticks(angles[:-1]); ax_rad.set_xticklabels(labels, size=7.5, color=TX)
    ax_rad.set_yticks([25, 50, 75]); ax_rad.set_yticklabels(["25","50","75"], size=5.5, color=MU)
    ax_rad.plot(angles, vals_pct, color=BL, lw=2)
    ax_rad.fill(angles, vals_pct, color=BL, alpha=0.25)
    ax_rad.set_title("Lig Percentile", size=8, color=MU, pad=8)

    # Metrik tablo
    ax_tbl = fig.add_axes([0.46, 0.40, 0.25, 0.50])
    ax_tbl.set_facecolor(S1); ax_tbl.axis("off"); _border(ax_tbl)
    ax_tbl.text(0.5, 0.97, "p90 Metrikler", ha="center", va="top",
                fontsize=9, color=TX, fontweight="bold", transform=ax_tbl.transAxes)
    for i, (label, val) in enumerate(p90.items()):
        y = 0.88 - i * 0.09
        pct_val = pct.get(label, 50)
        color = GR if pct_val >= 70 else (OR if pct_val >= 40 else RE)
        ax_tbl.text(0.06, y, label, va="top", fontsize=7.5, color=TX, transform=ax_tbl.transAxes)
        ax_tbl.text(0.92, y, f"{val:.2f}", va="top", fontsize=7.5, color=color,
                    ha="right", transform=ax_tbl.transAxes)

    # Benzer oyuncular
    ax_sim = fig.add_axes([0.73, 0.40, 0.25, 0.50])
    ax_sim.set_facecolor(S1); ax_sim.axis("off"); _border(ax_sim)
    ax_sim.text(0.5, 0.97, "Benzer Oyuncular", ha="center", va="top",
                fontsize=9, color=TX, fontweight="bold", transform=ax_sim.transAxes)

    if not all_players_df.empty and "player_name" in all_players_df.columns:
        try:
            # Geçici DataFrame ile basit cosine similarity
            feat_cols = ["gol","ast","xg","xa","sut","pas","intp","dw","prun"]
            avail = [c for c in feat_cols if c in all_players_df.columns]
            pool = all_players_df[all_players_df["mac"] >= 3].copy()
            if name in pool["player_name"].values and len(pool) > 1:
                X = pool[avail].fillna(0).astype(float)
                dk_s = pool["dk"].replace(0, np.nan)
                X = X.div(dk_s, axis=0).fillna(0) * 90
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import cosine_similarity
                Xs = StandardScaler().fit_transform(X)
                idx = pool[pool["player_name"] == name].index[0]
                pos_idx = pool.index.get_loc(idx)
                sims = cosine_similarity(Xs[pos_idx:pos_idx+1], Xs)[0]
                pool = pool.copy(); pool["_sim"] = sims
                top = pool[pool["player_name"] != name].nlargest(4, "_sim")
                for i, (_, row) in enumerate(top.iterrows()):
                    y = 0.87 - i * 0.12
                    ax_sim.text(0.06, y, row["player_name"], va="top", fontsize=7.5,
                                color=TX, transform=ax_sim.transAxes)
                    ax_sim.text(0.94, y - 0.04, f"{row['_sim']:.0%}",
                                va="top", fontsize=6.5, color=MU, ha="right",
                                transform=ax_sim.transAxes)
        except Exception:
            ax_sim.text(0.5, 0.5, "—", ha="center", color=MU, transform=ax_sim.transAxes)

    # Özet metin
    ax_sum = fig.add_axes([0.46, 0.08, 0.52, 0.28])
    ax_sum.set_facecolor(S1); ax_sum.axis("off"); _border(ax_sum)
    top2_strong = sorted(pct.items(), key=lambda x: -x[1])[:2]
    top2_weak   = sorted(pct.items(), key=lambda x: x[1])[:2]
    lines = []
    for lbl, v in top2_strong:
        lines.append(f"✅ {lbl} — lig içinde {v:.0f}. percentile (güçlü)")
    for lbl, v in top2_weak:
        lines.append(f"⚠️ {lbl} — lig içinde {v:.0f}. percentile (geliştirilebilir)")
    for i, line in enumerate(lines):
        y = 0.85 - i * 0.20
        color = GR if line.startswith("✅") else OR
        ax_sum.text(0.04, y, line, va="top", fontsize=8, color=color,
                    transform=ax_sum.transAxes)

    pdf.savefig(fig, dpi=120); plt.close(fig)


# ── Transfer hedef listesi sayfası ─────────────────────────────────────────
def _p_transfer_targets(pdf, team_name: str, player_pool, style_target: Optional[str] = None):
    """Kadrodaki eksik pozisyonlar + önerilen transfer hedefleri."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.96, f"Transfer Hedefleri — {team_name}", ha="center",
             fontsize=14, color=BL, fontweight="bold")
    if style_target:
        fig.text(0.5, 0.935, f"Stil Hedefi: {style_target}", ha="center",
                 fontsize=8.5, color=MU)

    if player_pool.empty:
        fig.text(0.5, 0.5, "Oyuncu havuzu yok.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    # Mevcut takım kadrosu
    team_df = player_pool[player_pool["team_name"].str.lower() == team_name.lower()]
    if team_df.empty:
        fig.text(0.5, 0.5, "Bu takım için oyuncu bulunamadı.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    # Basit kadro eksik tespiti
    pos_counts = team_df["position"].str.upper().str[:1].value_counts()
    targets_text = []
    needed = {"G": 2, "D": 4, "M": 4, "F": 2}
    for pos_code, req in needed.items():
        current = int(pos_counts.get(pos_code, 0))
        if current < req:
            gap = req - current
            targets_text.append((pos_code, gap))

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.85])
    ax.set_facecolor(S1); ax.axis("off"); _border(ax)

    if not targets_text:
        ax.text(0.5, 0.5, "Kadro tamamlanmış görünüyor — eksik pozisyon yok.",
                ha="center", va="center", fontsize=11, color=GR, transform=ax.transAxes)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    pos_labels = {"G": "Kaleci", "D": "Defans", "M": "Orta Saha", "F": "Forvet"}
    y = 0.93
    for pos_code, gap in targets_text:
        ax.text(0.03, y, f"📌 {pos_labels.get(pos_code, pos_code)} — {gap} eksik",
                va="top", fontsize=10, color=OR, fontweight="bold", transform=ax.transAxes)
        y -= 0.05

        # Takım dışı havuzdan öneri
        ext = player_pool[
            (player_pool["team_name"].str.lower() != team_name.lower()) &
            (player_pool["position"].str.upper().str[:1] == pos_code)
        ]
        if not ext.empty:
            sort_col = "rating" if "rating" in ext.columns else ext.columns[0]
            top5 = ext.nlargest(5, sort_col)[["player_name","team_name","position","rating","matches_played"]]
            for _, row in top5.iterrows():
                rat = f"{row.get('rating', 0):.2f}" if row.get('rating') else "—"
                line = f"  {row['player_name']} ({row['team_name']}) — {rat}"
                ax.text(0.05, y, line, va="top", fontsize=8, color=TX, transform=ax.transAxes)
                y -= 0.04
        y -= 0.02

    pdf.savefig(fig, dpi=120); plt.close(fig)


def generate_team_report(team_name: str, season: Optional[str] = None,
                         output_path: Optional[str] = None,
                         opponent: Optional[str] = None,
                         include_players: bool = False,
                         style_target: Optional[str] = None) -> str:
    _rc()
    d = _load(team_name, season)
    if not d:
        raise ValueError(f"'{team_name}' için veritabanında maç bulunamadı.")
    if output_path is None:
        safe = team_name.lower().replace(" ", "_")
        output_path = str(Path(tempfile.gettempdir()) / f"{safe}_rapor.pdf")
    ins = _insights(d)

    # Takım istatistikleri (stil analizi için)
    try:
        import pandas as pd
        team_stats_df = db.get_team_stats(None, season)
    except Exception:
        import pandas as pd
        team_stats_df = pd.DataFrame()

    # Rakip verisi (H2H için)
    d_opp = None
    if opponent:
        d_opp = _load(opponent, season)

    # Oyuncu havuzu (include_players + transfer hedefleri için)
    try:
        from src.api.main import _load_player_pool as _lpp
        player_pool = _lpp(league=None, season=season, min_minutes=90, limit=3000)
    except Exception:
        import pandas as pd
        player_pool = pd.DataFrame()

    with PdfPages(output_path) as pdf:
        _p_cover(pdf, d)
        _p_results(pdf, d)
        _p_shotmaps(pdf, d)
        _p_shot_overlay(pdf, d)
        _p_xg(pdf, d)
        _p_players(pdf, d)
        _p_insights(pdf, d, ins)

        # "Neden Başarılı?" sayfası
        _p_why_successful(pdf, d, team_stats_df)

        # H2H sayfası (rakip veriliyse)
        if d_opp:
            _p_h2h_tactical(pdf, d, d_opp, team_stats_df)

        # Oyuncu detay sayfaları
        if include_players and not d["players"].empty:
            for _, p_row in d["players"].iterrows():
                _p_player_detail(pdf, p_row.to_dict(), d["players"])

        # Transfer hedefleri sayfası
        _p_transfer_targets(pdf, team_name, player_pool, style_target=style_target)

    return output_path


def generate_comparison_report(team_a:str, team_b:str, season:Optional[str]=None,
                                output_path:Optional[str]=None)->str:
    _rc()
    da=_load(team_a,season); db2=_load(team_b,season)
    if not da: raise ValueError(f"'{team_a}' için maç bulunamadı.")
    if not db2: raise ValueError(f"'{team_b}' için maç bulunamadı.")
    if output_path is None:
        sa=team_a.lower().replace(" ","_"); sb=team_b.lower().replace(" ","_")
        output_path=str(Path(tempfile.gettempdir())/f"{sa}_vs_{sb}_rapor.pdf")
    with PdfPages(output_path) as pdf:
        _p_h2h_cover(pdf,da,db2)
        _p_h2h_metrics(pdf,da,db2)
        _p_h2h_shots(pdf,da,db2)
        _p_h2h_strategy(pdf,da,db2)
    return output_path
