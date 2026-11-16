"""Team performance report and head-to-head comparison report generator."""

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
    """Draw a visible border around the axes (works after axis("off") too)."""
    ax.add_patch(plt.Rectangle((0,0),1,1,transform=ax.transAxes,
                                fill=False,edgecolor=color,linewidth=lw,
                                clip_on=False,zorder=20))

def _cap(ax, text):
    """Add an italic caption line along the bottom edge inside the panel."""
    ax.text(0.01, 0.013, text, transform=ax.transAxes,
            fontsize=5.8, color=MU, ha="left", va="bottom",
            clip_on=True, style="italic", wrap=False,
            bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.75),
            zorder=25)

# ── Pitch drawing (coordinates: x = depth 0-52, y = width 0-100) ───
def _half_pitch(ax, bg=S1):
    ax.set_facecolor(bg); ax.set_xlim(-3,55); ax.set_ylim(-3,103)
    ax.set_aspect("equal"); ax.axis("off")
    kw=dict(color=BR,lw=1.2,zorder=1)
    # Penalty area
    ax.plot([0,16.5],[21.1,21.1],**kw); ax.plot([0,16.5],[78.9,78.9],**kw)
    ax.plot([16.5,16.5],[21.1,78.9],**kw)
    # Six-yard box
    ax.plot([0,5.5],[36.8,36.8],**kw); ax.plot([0,5.5],[63.2,63.2],**kw)
    ax.plot([5.5,5.5],[36.8,63.2],**kw)
    # Kale
    ax.plot([0,-2.44],[45.2,45.2],color=BL,lw=2,zorder=2)
    ax.plot([0,-2.44],[54.8,54.8],color=BL,lw=2,zorder=2)
    ax.plot([-2.44,-2.44],[45.2,54.8],color=BL,lw=2,zorder=2)
    # Penalty spot + arc
    ax.plot(11,50,"o",color=BR,ms=2.5,zorder=2)
    ax.add_patch(Arc((11,50),18.3,18.3,angle=0,theta1=-53,theta2=53,color=BR,lw=1.2))
    # Halfway line (half-pitch boundary)
    ax.plot([52,52],[0,100],**kw)
    # Pitch background rectangle
    ax.add_patch(plt.Rectangle((0,0),52,100,fill=True,facecolor=bg,edgecolor=BR,lw=1.2,zorder=0))

def _goal_frame(ax):
    """Goal mouth seen from the front — placement map of shots & saves.
    set_aspect removed; filling the cell keeps it readable."""
    ax.set_facecolor(S2)
    ax.set_xlim(33, 67)        # goal posts: 36.8 – 63.2  (±3 margin)
    ax.set_ylim(-0.6, 3.6)    # crossbar: 2.44m          (bottom/top margin)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(BR); s.set_linewidth(1.2)

    # Net background (inside the goal)
    ax.add_patch(plt.Rectangle((36.8,0),26.4,2.44,
                                facecolor="#0f1f0f",edgecolor="none",alpha=0.7,zorder=2))
    # Grass
    ax.add_patch(plt.Rectangle((33,-0.6),34,0.6,
                                facecolor="#1a2e1a",edgecolor="none",alpha=0.6,zorder=1))
    ax.axhline(0,color="#2e4a2e",lw=5,alpha=0.5,zorder=3)

    # Posts + crossbar
    kw=dict(color="#e0e0e0",lw=2.5,zorder=6)
    ax.plot([36.8,36.8],[0,2.44],**kw)
    ax.plot([63.2,63.2],[0,2.44],**kw)
    ax.plot([36.8,63.2],[2.44,2.44],**kw)

    # Net grid
    for x in np.linspace(36.8,63.2,8)[1:-1]:
        ax.plot([x,x],[0,2.44],color=BR,lw=0.5,alpha=0.45,zorder=4)
    for y in np.linspace(0,2.44,6)[1:-1]:
        ax.plot([36.8,63.2],[y,y],color=BR,lw=0.5,alpha=0.45,zorder=4)

    # Zone labels (left / centre / right)
    for xi,lbl in [(39.5,"Left"),(50,"Centre"),(60.5,"Right")]:
        ax.text(xi,-0.38,lbl,ha="center",va="top",fontsize=6,color=MU,zorder=7)
    # Height labels
    for yi,lbl in [(0.4,"Low"),(1.22,"Mid"),(2.1,"High")]:
        ax.text(34.2,yi,lbl,ha="right",va="center",fontsize=5.5,color=MU,zorder=7)

# ── Data loader ─────────────────────────────────────────────────────────────
def _load(team:str, season:Optional[str]=None)->dict:
    slug=f"%{team}%"; p={"slug":slug}
    # sf = filter for the matches table (no alias), sf2 = for JOINed queries
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

# ── Heatmap (seaborn KDE + zone analysis) ──────────────────────────────────
def _heatmap(ax, df, title, colors, label_zones=True):
    _half_pitch(ax, bg=BG)
    ax.set_title(title, fontsize=9, color=TX, pad=5, loc="left")
    df=df.dropna(subset=["player_coord_x","player_coord_y"])
    if len(df)<5: return

    x=df["player_coord_x"].clip(0,50).values
    y=df["player_coord_y"].clip(0,100).values

    # Scipy KDE → contourf (seaborn kdeplot can misbehave when clip is unsupported)
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
        # Share per zone in %
        zones=[
            ("Penalty\nArea", (0,16.5),(21.1,78.9)),
            ("Central\nAttack",  (16.5,28),(30,70)),
            ("Long\nRange",   (28,52),(10,90)),
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

# ── Shot map ────────────────────────────────────────────────────────────────
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
    # xG total — positioned in axes fraction (does not clash with the legend)
    total_xg=df["xg"].sum() if "xg" in df.columns else 0
    goals_cnt=(df["shot_type"]=="goal").sum()
    ax.text(0.02,0.97,f"xG {total_xg:.1f} | {int(goals_cnt)} goals | {len(df)} shots",
            transform=ax.transAxes,fontsize=7,color=MU,ha="left",va="top",zorder=15)

# ── Auto insights ────────────────────────────────────────────────────────────
def _insights(d:dict)->list:
    shots,shots_ag,matches,players=d["shots"],d["shots_ag"],d["matches"],d["players"]
    GA=d["GA"]; ins=[]

    if not shots.empty:
        xgt=shots["xg"].sum(); gg=(shots["shot_type"]=="goal").sum()
        diff=gg-xgt
        if diff<-3: ins.append(("⚠️","Finishing Crisis",
            f"xG {xgt:.1f} but {int(gg)} goals — {diff:.1f} chances wasted."))
        elif diff>3: ins.append(("✅","Clinical Finishing",
            f"xG {xgt:.1f} but {int(gg)} goals — above expectation."))
        far=(shots["player_coord_x"]>=25).sum()/max(len(shots),1)
        if far>0.25: ins.append(("⚠️","Too Many Long Shots",
            f"{far*100:.0f}% of shots from 25m+."))
        dt=shots[shots["situation"].isin(["set-piece","throw-in-set-piece","corner"])]
        if len(dt)/max(len(shots),1)>0.22: ins.append(("✅","Set-Piece Threat",
            f"{len(dt)/len(shots)*100:.0f}% of shots from set pieces."))

    ev=matches[matches["side"]=="home"]; dep=matches[matches["side"]=="away"]
    ep=ev["res"].map({"G":3,"B":1,"M":0}).sum()
    dp=dep["res"].map({"G":3,"B":1,"M":0}).sum()
    if ep-dp>8: ins.append(("⚠️","Weak Away Form",
        f"{ep} points at home, only {dp} away."))
    if GA/max(len(matches),1)>1.1: ins.append(("⚠️","Defensive Gaps",
        f"{GA/len(matches):.2f} goals conceded per match."))
    if not shots_ag.empty and len(shots_ag)/max(len(matches),1)>12:
        ins.append(("⚠️","High Shot Pressure",
            f"{len(shots_ag)/len(matches):.1f} opponent shots per match."))

    if not players.empty:
        b=players[players["mac"]>=8].dropna(subset=["rat"]).nlargest(1,"rat")
        if not b.empty:
            r=b.iloc[0]
            ins.append(("⭐","Best Player",
                f"{r['player_name']} ({r['position']}) — {r['rat']:.2f} rating in {r['mac']} matches."))
        t=players.nlargest(1,"gol")
        if not t.empty and t.iloc[0]["gol"]>0:
            r=t.iloc[0]
            ins.append(("⚽","Top Scorer",
                f"{r['player_name']} — {int(r['gol'])} goals / {r['xg']:.1f} xG."))

    s5=matches.tail(5)["res"].tolist(); s5p=s5.count("G")*3+s5.count("B")
    if s5p<=4: ins.append(("📉","Poor Form",f"{s5p} points in the last 5 matches ({''.join({'G':'W','B':'D','M':'L'}[x] for x in s5)})."))
    elif s5p>=10: ins.append(("📈","Good Form",f"{s5p} points in the last 5 matches ({''.join({'G':'W','B':'D','M':'L'}[x] for x in s5)})."))
    return ins

# ── Page helpers ────────────────────────────────────────────────────────────
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
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _p_cover(pdf, d:dict):
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    seas=d["matches"]["season"].iloc[0] if not d["matches"].empty else ""
    lg=d["matches"]["league"].iloc[0] if not d["matches"].empty else ""

    # Background gradient strip
    grad=np.linspace(0,1,256).reshape(1,-1)
    ax.imshow(grad,extent=[0,16,9.5,11],aspect="auto",cmap="Blues",alpha=0.08,zorder=0)

    fig.text(0.5,0.73,d["team_name"].upper(),ha="center",fontsize=52,
             fontweight="bold",color=TX,transform=fig.transFigure)
    fig.text(0.5,0.64,f"{seas}  ·  {lg}",ha="center",fontsize=16,color=BL,
             transform=fig.transFigure)
    fig.text(0.5,0.57,"Comprehensive Performance Report",ha="center",fontsize=12,color=MU,
             transform=fig.transFigure)

    # Divider line
    fig.add_artist(plt.Line2D([0.1,0.9],[0.54,0.54],transform=fig.transFigure,
                              color=BL,lw=1,alpha=0.4))

    # Stat boxes
    W,D,L=d["W"],d["D"],d["L"]
    stats=[
        (str(len(d["matches"])),"Matches",BL),
        (f"{W}W {D}D {L}L","W-D-L",GR if W>L else RE),
        (str(d["PTS"]),"Points",OR),
        (f"{d['GF']}-{d['GA']}","GF-GA",YE),
        (str(len(d["shots"])),"Shots",PU),
        (f"{d['TXG']:.1f}","Total xG",OR),
    ]
    xs=np.linspace(0.08,0.80,6)
    for i,((v,lbl,c),x) in enumerate(zip(stats,xs)):
        _stat_box(fig,x,0.36,v,lbl,color=c)

    # Form strip
    strip=_form_strip(d["matches"],10)
    fig.text(0.5,0.30,"Last 10 Matches",ha="center",fontsize=9,color=MU,
             transform=fig.transFigure)
    sw=0.038; gap=0.004; total=len(strip)*(sw+gap)
    sx0=0.5-total/2
    for i,(res,c,opp) in enumerate(strip):
        x=sx0+i*(sw+gap)
        fig.add_artist(plt.Rectangle((x,0.22),sw,0.055,transform=fig.transFigure,
                                      facecolor=c,edgecolor="none",alpha=0.85))
        fig.text(x+sw/2,0.247+0.003,res,ha="center",va="center",fontsize=10,
                 fontweight="bold",color=BG,transform=fig.transFigure)

    fig.text(0.5,0.06,"Footscrapping Analytics · Match Data",ha="center",
             fontsize=8,color=MU,transform=fig.transFigure)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_results(pdf, d:dict):
    matches=d["matches"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.97,"Results & Form Analysis",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.35,
                top=0.91,bottom=0.07,left=0.07,right=0.97)

    # Score bars
    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor(S1)
    ax1.set_title(f"Weekly Score  —  {d['team_name']} (dark) vs Opponent (faded)",
                  fontsize=9,color=TX,loc="left",pad=5)
    n=len(matches); xi=np.arange(n)
    kg=matches["kg"].fillna(0).values; og=matches["og"].fillna(0).values
    for i,(res,k,o) in enumerate(zip(matches["res"],kg,og)):
        c={"G":GR,"B":OR,"M":RE}[res]
        ax1.bar(i-0.22,k,0.4,color=c,alpha=0.9,zorder=3)
        ax1.bar(i+0.22,o,0.4,color=MU,alpha=0.4,zorder=3)
        ax1.text(i,max(k,o)+0.12,{"G":"W","B":"D","M":"L"}[res],ha="center",fontsize=6.5,color=c,
                 fontweight="bold",zorder=4)
    xtl=matches["round"].fillna("?").astype(str).tolist()
    ax1.set_xticks(xi); ax1.set_xticklabels(xtl,fontsize=6.5,rotation=0)
    ax1.set_xlabel("Matchweek",color=MU,fontsize=8); ax1.set_ylabel("Goals",color=MU,fontsize=8)
    ax1.grid(axis="y",alpha=0.25,zorder=0); _sp(ax1); _border(ax1)
    _cap(ax1,"Weekly scores. Coloured bar = team goals, faded bar = opponent goals. The letter on top is the result (W/D/L).")

    # Cumulative points
    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(S1)
    ax2.set_title("Cumulative Points",fontsize=9,color=TX,loc="left",pad=5)
    pts_s=matches["res"].map({"G":3,"B":1,"M":0}).cumsum()
    r2=range(1,n+1); ideal=np.array(r2)*1.5
    ax2.plot(r2,pts_s.values,color=BL,lw=2.5,zorder=4)
    ax2.plot(r2,ideal,color=BR,lw=1.5,ls="--",zorder=3)
    ax2.fill_between(r2,pts_s.values,ideal,where=pts_s.values>=ideal,alpha=0.12,color=GR,zorder=2)
    ax2.fill_between(r2,pts_s.values,ideal,where=pts_s.values<ideal,alpha=0.12,color=RE,zorder=2)
    ax2.legend(["Actual","1.5 pts/match"],facecolor=S2,edgecolor=BR,fontsize=7)
    ax2.grid(alpha=0.25); _sp(ax2); _border(ax2)
    _cap(ax2,"Points accumulated over the season. Dashed line = 1.5 points per match. Green shows spells above that pace, red spells below it.")

    # Home/Away
    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(S1)
    ax3.set_title("Home / Away",fontsize=9,color=TX,loc="left",pad=5)
    em=matches[matches["side"]=="home"]; dm=matches[matches["side"]=="away"]
    cats=["W","D","L","GF","GA"]
    ev_v=[(em["res"]=="G").sum(),(em["res"]=="B").sum(),(em["res"]=="M").sum(),
          em["kg"].sum(),em["og"].sum()]
    dp_v=[(dm["res"]=="G").sum(),(dm["res"]=="B").sum(),(dm["res"]=="M").sum(),
          dm["kg"].sum(),dm["og"].sum()]
    xi2=np.arange(5)
    ax3.bar(xi2-0.22,ev_v,0.42,color=BL,alpha=0.85,label=f"Home ({len(em)})")
    ax3.bar(xi2+0.22,dp_v,0.42,color=PU,alpha=0.85,label=f"Away ({len(dm)})")
    for i,(e,d2) in enumerate(zip(ev_v,dp_v)):
        ax3.text(i-0.22,e+0.15,str(int(e)),ha="center",fontsize=8,color=TX)
        ax3.text(i+0.22,d2+0.15,str(int(d2)),ha="center",fontsize=8,color=TX)
    ax3.set_xticks(xi2); ax3.set_xticklabels(cats,fontsize=9)
    ax3.legend(facecolor=S2,edgecolor=BR,fontsize=7)
    ax3.grid(axis="y",alpha=0.25); _sp(ax3); _border(ax3)
    _cap(ax3,"Wins/draws/losses and goals split by home and away matches. Shows any home advantage or disadvantage.")

    # Son 10 rakip tablosu
    ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(S1); ax4.axis("off")
    ax4.set_title("Recent Matches",fontsize=9,color=TX,loc="left",pad=5)
    last=matches.tail(10)[["opp","kg","og","res"]].copy()
    y_pos=0.93
    ax4.text(0.02,y_pos,"Opponent",transform=ax4.transAxes,fontsize=7.5,
             color=MU,fontweight="bold")
    ax4.text(0.72,y_pos,"Score",transform=ax4.transAxes,fontsize=7.5,
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
        ax4.text(0.92,y_pos,{"G":"W","B":"D","M":"L"}.get(row["res"],row["res"]),transform=ax4.transAxes,
                 fontsize=7.5,color=c,fontweight="bold",ha="center")
        y_pos-=0.085
    _sp(ax4); _border(ax4)
    _cap(ax4,"Opponent, score and result of the last 10 matches. A quick read of current form and opposition quality.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_shotmaps(pdf, d:dict):
    # vertical figsize (11×20) → pitch ratio (58:106≈0.547) fits the cells exactly
    fig=plt.figure(figsize=(11,20)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.985,"Shot & Density Maps",ha="center",
             fontsize=13,fontweight="bold",color=TX)

    gs=GridSpec(2,2,figure=fig,hspace=0.14,wspace=0.10,
                top=0.975,bottom=0.025,left=0.03,right=0.97)

    ax1=fig.add_subplot(gs[0,0])
    _shotmap(ax1,d["shots"],f"⚽  {d['team_name']} — Shots For"); _border(ax1)
    _cap(ax1,"Location of every shot by the team. Dot size is proportional to xG; ★ goal, circle = saved/missed shot.")

    ax2=fig.add_subplot(gs[0,1])
    _shotmap(ax2,d["shots_ag"],"🛡️  Opponent Shots (Defence)"); _border(ax2)
    _cap(ax2,"Location of shots conceded. Shows which areas of the defensive zone came under pressure.")

    ax3=fig.add_subplot(gs[1,0])
    _heatmap(ax3,d["shots"],"🔥  Attacking Density",[OR,"#ff4500","#ff0000"]); _border(ax3)
    _cap(ax3,"KDE density map of shots for. Warm colours mark where most shots come from; % values are each zone's share of the total.")

    ax4=fig.add_subplot(gs[1,1])
    _heatmap(ax4,d["shots_ag"],"🔵  Defensive Pressure Density",[BL,PU,"#8800ff"]); _border(ax4)
    _cap(ax4,"Density of opponent shots. Purple/blue zones are where the defence struggled most — potential weak spots.")

    fig.text(0.5,0.010,
             "Size = xG  ·  ★ = goal  ·  % = share of shots in the zone",
             ha="center",fontsize=7.5,color=MU)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _shot_overlay(ax, df, title, colors):
    """Draw the KDE heatmap as background and shot points on top."""
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
    ax.text(0.02, 0.97, f"xG {total_xg:.1f} | {int(goals_cnt)} goals | {len(df)} shots",
            transform=ax.transAxes, fontsize=7, color=MU, ha="left", va="top", zorder=15)


def _p_shot_overlay(pdf, d: dict):
    """Shot map overlaid on the density map."""
    fig = plt.figure(figsize=(11, 12)); fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.985, "Shots + Density Overlay",
             ha="center", fontsize=13, fontweight="bold", color=TX)
    gs = GridSpec(1, 2, figure=fig, hspace=0.10, wspace=0.10,
                  top=0.965, bottom=0.03, left=0.03, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    _shot_overlay(ax1, d["shots"], f"⚽  {d['team_name']} — Attack", [OR, "#ff4500", "#ff0000"])
    _border(ax1)
    _cap(ax1, "Shot locations over the density map. Heat → where most shots are produced; ★ goals.")

    ax2 = fig.add_subplot(gs[0, 1])
    _shot_overlay(ax2, d["shots_ag"], "🛡️  Opponent — Defence", [BL, PU, "#8800ff"])
    _border(ax2)
    _cap(ax2, "Opponent shot density + shot locations. Red/purple = where the defence struggled most.")

    pdf.savefig(fig, bbox_inches="tight", facecolor=BG); plt.close(fig)


def _p_xg(pdf, d:dict):
    shots,shots_ag,xgd=d["shots"],d["shots_ag"],d["xgd"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Goal Mouth & xG Analysis",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,
                top=0.92,bottom=0.07,left=0.06,right=0.97)

    def gm_plot(ax, df, title):
        _goal_frame(ax)
        ax.set_title(title, fontsize=9, color=TX, pad=4, loc="left")
        df = df.dropna(subset=["goal_mouth_y"])
        if df.empty: return
        # goal_mouth_x is not populated in the DB; xgot is used as a height proxy
        # (high xgot → shot placed in a corner/hard spot → drawn higher)
        rng = np.random.default_rng(0)
        def xy(row):
            gx = 36.8 + (row["goal_mouth_y"] / 100) * (63.2 - 36.8)
            xgot_val = float(row.get("xgot") or 0.0)
            # xgot 0-1 → goal height 0-2.44 + small deterministic jitter
            gy = np.clip(xgot_val * 2.44, 0.05, 2.39) + rng.uniform(-0.18, 0.18)
            return gx, float(np.clip(gy, 0.02, 2.42))
        saves = df[df["shot_type"] == "save"]
        goals = df[df["shot_type"] == "goal"]
        if not saves.empty:
            xs, ys = zip(*[xy(r) for _, r in saves.iterrows()])
            ax.scatter(xs, ys, c=BL, s=55, alpha=0.65, edgecolors=BG, lw=0.5, zorder=8, label="Save")
        if not goals.empty:
            xg2, yg = zip(*[xy(r) for _, r in goals.iterrows()])
            ax.scatter(xg2, yg, c=GR, s=130, alpha=0.95, marker="*",
                       edgecolors="white", lw=0.7, zorder=10, label="Goal")
            for xi, yi, (_, row) in zip(xg2, yg, goals.iterrows()):
                if row.get("xg"):
                    ax.text(xi, yi + 0.20, f"{row['xg']:.2f}", ha="center",
                            fontsize=5.5, color=YE, zorder=11)
        ax.legend(loc="upper right", facecolor=S2, edgecolor=BR, fontsize=7,
                  framealpha=0.9)
        _border(ax)

    ax1=fig.add_subplot(gs[0,0]); gm_plot(ax1,shots,"⚽  Goals Scored")
    _cap(ax1,"Placement of goals scored within the goal. ★ goal, circle = save. The yellow number next to it is the xG of the chance (how hard the goal was).")
    ax2=fig.add_subplot(gs[0,1]); gm_plot(ax2,shots_ag,"🧤  Goals Conceded")
    _cap(ax2,"Placement of goals conceded. Shows which corner and height troubled the goalkeeper most.")

    # xG vs goals per match
    ax3=fig.add_subplot(gs[0,2]); ax3.set_facecolor(S1)
    ax3.set_title("xG vs Goals (per match)",fontsize=9,color=TX,loc="left",pad=4)
    if not xgd.empty:
        r3=range(1,len(xgd)+1)
        ax3.bar([i-0.22 for i in r3],xgd["xg"],0.42,color=OR,alpha=0.8,label="xG")
        ax3.bar([i+0.22 for i in r3],xgd["gol"],0.42,color=GR,alpha=0.8,label="Goals")
        ax3_r=ax3.twinx()
        cumxg=xgd["xg"].cumsum(); cumg=xgd["gol"].cumsum()
        ax3_r.plot(r3,(cumg-cumxg).values,color=RE,lw=1.5,ls="--",alpha=0.7)
        ax3_r.axhline(0,color=BR,lw=0.8,ls=":",alpha=0.5)
        ax3_r.set_ylabel("Cumulative G-xG",color=RE,fontsize=7)
        ax3_r.tick_params(axis="y",colors=RE,labelsize=7)
        ax3.legend(facecolor=S2,edgecolor=BR,fontsize=7)
        ax3.grid(axis="y",alpha=0.25)
    _sp(ax3); _border(ax3)
    _cap(ax3,"Expected goals (orange) vs actual goals (green) per match. Dashed red = cumulative G−xG; positive = clinical finishing.")

    # Shot quality
    ax4=fig.add_subplot(gs[1,0]); ax4.set_facecolor(S1)
    ax4.set_title("Shot Quality (xG band)",fontsize=9,color=TX,loc="left",pad=4)
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
        ax4.set_xlabel("Shots",color=MU,fontsize=8)
        ax4.grid(axis="x",alpha=0.25)
    _sp(ax4); _border(ax4)
    _cap(ax4,"Shots by xG band. Green/blue rows (35%+) are high-quality chances; red/grey ones are low quality.")

    # Durum analizi
    ax5=fig.add_subplot(gs[1,1]); ax5.set_facecolor(S1)
    ax5.set_title("Situation",fontsize=9,color=TX,loc="left",pad=4)
    if not shots.empty:
        nm={"regular":"Open play","assisted":"Assisted","set-piece":"Set piece",
            "corner":"Corner","throw-in-set-piece":"Throw-in","counter-attack":"Counter"}
        sit=shots["situation"].map(nm).fillna(shots["situation"]).value_counts().head(6)
        ax5.barh(sit.index,sit.values,color=BL,alpha=0.82,edgecolor=BG,lw=0.5)
        for i,v in enumerate(sit.values):
            ax5.text(v+0.3,i,str(int(v)),va="center",fontsize=8,color=TX)
        ax5.grid(axis="x",alpha=0.25)
    _sp(ax5); _border(ax5)
    _cap(ax5,"Game situation the shots came from. Splits threat between set pieces, corners, open play and counter-attacks.")

    # Body part
    ax6=fig.add_subplot(gs[1,2]); ax6.set_facecolor(S1)
    ax6.set_title("Body Part",fontsize=9,color=TX,loc="left",pad=4)
    if not shots.empty:
        nm2={"right-foot":"Right foot","left-foot":"Left foot","head":"Head"}
        bp=shots["body_part"].map(nm2).fillna(shots["body_part"]).value_counts()
        ax6.bar(bp.index,bp.values,color=[GR,OR,BL,PU][:len(bp)],alpha=0.85,edgecolor=BG,lw=0.5)
        for b in ax6.patches:
            ax6.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,
                     str(int(b.get_height())),ha="center",fontsize=9,color=TX)
        ax6.grid(axis="y",alpha=0.25)
    _sp(ax6); _border(ax6)
    _cap(ax6,"Body part used for shots. The right/left foot split shows attacking preference; a high header share means reliance on crosses.")
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_players(pdf, d:dict):
    players=d["players"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Player Performance Analysis",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,
                top=0.92,bottom=0.07,left=0.08,right=0.97)

    p5=players[players["mac"]>=5].copy().fillna(0)

    # Rating bars
    ax1=fig.add_subplot(gs[0,:]); ax1.set_facecolor(S1)
    ax1.set_title("Rating Ranking (min 5 matches)",fontsize=9,color=TX,loc="left",pad=5)
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
    lp=[mpatches.Patch(color=YE,label="GK"),mpatches.Patch(color=BL,label="DF"),
        mpatches.Patch(color=GR,label="MF"),mpatches.Patch(color=RE,label="FW")]
    ax1.legend(handles=lp,facecolor=S2,edgecolor=BR,fontsize=8,loc="lower right",
               title="Position",title_fontsize=7,ncol=4)
    ax1.grid(axis="x",alpha=0.25); _sp(ax1); _border(ax1)
    _cap(ax1,"Season average of the match rating (min 5 matches). The dashed 6.5 line represents the league average. Colour = position.")

    # xG vs Gol scatter
    ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor(S1)
    ax2.set_title("Goals vs xG (individual)",fontsize=9,color=TX,loc="left",pad=4)
    forv=p5[p5["xg"]>0].copy()
    if not forv.empty:
        ax2.scatter(forv["xg"],forv["gol"],c=[_pc(r) for r in forv["position"]],
                    s=forv["mac"]*8,alpha=0.82,edgecolors="white",lw=0.4,zorder=5)
        mv=max(forv["xg"].max(),forv["gol"].max())+0.5
        ax2.plot([0,mv],[0,mv],color=BR,ls="--",lw=1)
        for _,row in forv[forv["xg"]>1.5].iterrows():
            ax2.annotate(row["player_name"].split()[-1],(row["xg"],row["gol"]),
                         fontsize=6.5,color=TX,xytext=(4,4),textcoords="offset points")
        ax2.set_xlabel("xG",color=MU,fontsize=8); ax2.set_ylabel("Goals",color=MU,fontsize=8)
        ax2.text(0.03,0.92,"↑ above xG",transform=ax2.transAxes,fontsize=7,color=GR)
        ax2.text(0.55,0.05,"↓ below xG",transform=ax2.transAxes,fontsize=7,color=RE)
        ax2.grid(alpha=0.25)
    _sp(ax2); _border(ax2)
    _cap(ax2,"Expected goals (xG) vs actual goals per player. Above the diagonal = clinical finishing, below = missed chances. Dot size is proportional to matches played.")

    # Goal contributions p90
    ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor(S1)
    ax3.set_title("Goal Contributions p90 (min 300 min)",fontsize=9,color=TX,loc="left",pad=4)
    c3=p5[p5["dk"]>=300].copy()
    if not c3.empty:
        c3["gkp90"]=(c3["gol"]+c3["ast"])/c3["dk"]*90
        c3=c3.nlargest(12,"gkp90")
        bars=ax3.barh(c3["player_name"],c3["gkp90"],
                      color=[_pc(r) for r in c3["position"]],alpha=0.85,edgecolor=BG,lw=0.4)
        for bar,v in zip(bars,c3["gkp90"]):
            ax3.text(v+0.003,bar.get_y()+bar.get_height()/2,f"{v:.2f}",
                     va="center",fontsize=7.5,color=TX)
        ax3.set_xlabel("(Goals+Assists)/90 min",color=MU,fontsize=8)
        ax3.grid(axis="x",alpha=0.25)
    _sp(ax3); _border(ax3)
    _cap(ax3,"(Goals + assists) normalised to 90 minutes. Compares the most effective attackers regardless of playing time.")

    # Anahtar pas
    ax4=fig.add_subplot(gs[1,2]); ax4.set_facecolor(S1)
    ax4.set_title("Key Passes (total)",fontsize=9,color=TX,loc="left",pad=4)
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
    _cap(ax4,"Total key passes (last pass before a shot). Identifies playmakers and creative wingers; measures chance creation even without assists.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_insights(pdf, d:dict, insights:list):
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,"Automatic Performance Assessment",ha="center",
             fontsize=13,fontweight="bold",color=TX)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    W,D,L,PTS,GF,GA=d["W"],d["D"],d["L"],d["PTS"],d["GF"],d["GA"]
    summ=(f"  {len(d['matches'])} Matches  ·  {W}W {D}D {L}L  ·  {PTS} Points  ·  "
          f"GF:{GF} GA:{GA}  ·  xG:{d['TXG']:.1f}  ·  Shots:{len(d['shots'])}")
    fig.text(0.5,0.915,summ,ha="center",fontsize=10,color=TX,
             bbox=dict(boxstyle="round,pad=0.5",facecolor=S2,edgecolor=BL,lw=1.5),
             transform=fig.transFigure)

    goods=[i for i in insights if i[0] in("✅","⭐","📈","⚽")]
    bads =[i for i in insights if i[0] in("⚠️","📉")]

    fig.add_artist(plt.Rectangle((0.03,0.10),0.455,0.76,transform=fig.transFigure,
                                  facecolor=S1,edgecolor=f"#1a4a2a",lw=1.5))
    fig.add_artist(plt.Rectangle((0.515,0.10),0.455,0.76,transform=fig.transFigure,
                                  facecolor=S1,edgecolor=f"#4a1a1a",lw=1.5))

    fig.text(0.06,0.835,"💪  STRENGTHS",fontsize=11,fontweight="bold",color=GR,
             transform=fig.transFigure)
    fig.text(0.545,0.835,"⚠️  AREAS TO IMPROVE",fontsize=11,fontweight="bold",
             color=RE,transform=fig.transFigure)

    y=0.79
    for ico,title,desc in (goods or [("—","No data","Not enough data.")]):
        fig.text(0.06,y,f"{ico}  {title}",fontsize=9.5,fontweight="bold",color=GR,
                 transform=fig.transFigure); y-=0.042
        for chunk in [desc[i:i+72] for i in range(0,len(desc),72)]:
            fig.text(0.07,y,chunk,fontsize=8,color=TX,transform=fig.transFigure)
            y-=0.028
        y-=0.01

    y=0.79
    for ico,title,desc in (bads or [("—","No data","Not enough data.")]):
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
# COMPARISON PAGES
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
        tips.append(f"✅ {ta} creates high xG ({a_xg:.2f}/match).\n   {tb}'s defence is open to pressure ({b_xga:.2f} xGA/match). Look for goals in quick transitions.")
    a_pos=_ts_avg(da,"ball_possession"); b_pos=_ts_avg(db2,"ball_possession")
    if a_pos>b_pos+6:
        tips.append(f"✅ {ta} dominate possession ({a_pos:.0f}% vs {b_pos:.0f}%).\n   Control the tempo and squeeze the opponent with a high block.")
    if not da["shots"].empty:
        sp=da["shots"]["situation"].isin(["set-piece","corner","throw-in-set-piece"]).mean()
        if sp>0.20:
            tips.append(f"✅ {ta} are strong from set pieces ({sp*100:.0f}%).\n   Prepare corners and free kicks specifically.")
    if not db2["shots_ag"].empty:
        b_box=(db2["shots_ag"]["player_coord_x"]<16.5).mean()
        if b_box>0.35:
            tips.append(f"⚠️ Be careful with shots from inside the box against {tb}.\n   They allow {b_box*100:.0f}% of shots from this zone.")
    dep_b=db2["matches"][db2["matches"]["side"]=="away"]
    if not dep_b.empty:
        dp=dep_b["res"].map({"G":3,"B":1,"M":0}).mean()
        if dp<1.0:
            tips.append(f"✅ {tb} are weak away (avg. {dp:.1f} points/match).\n   Depending on the venue, press for an early goal.")
    if not tips:
        tips.append("No significant statistical difference found.\nMore match data will deepen the analysis.")
    return tips


def _p_h2h_cover(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(BG); ax.axis("off")

    # Title
    fig.text(0.5,0.91,"COMPARISON REPORT",ha="center",fontsize=14,
             color=MU,fontweight="bold",transform=fig.transFigure)
    fig.text(0.28,0.82,ta.upper(),ha="center",fontsize=30,fontweight="bold",
             color=BL,transform=fig.transFigure)
    fig.text(0.5,0.82,"VS",ha="center",fontsize=22,color=MU,
             fontweight="bold",transform=fig.transFigure)
    fig.text(0.72,0.82,tb.upper(),ha="center",fontsize=30,fontweight="bold",
             color=RE,transform=fig.transFigure)

    fig.add_artist(plt.Line2D([0.1,0.9],[0.79,0.79],transform=fig.transFigure,
                              color=BR,lw=0.8))

    # Side-by-side stat grid
    def side_stats(d,xbase,color):
        W,D,L=d["W"],d["D"],d["L"]
        stats=[
            (str(len(d["matches"])),"Matches"),
            (f"{W}W {D}D {L}L","Record"),
            (str(d["PTS"]),"Points"),
            (f"{d['GF']}-{d['GA']}","GF-GA"),
            (f"{d['TXG']:.1f}","xG"),
            (str(len(d["shots"])),"Shots"),
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

    # Form strips
    for d,xbase,color in [(da,0.085,BL),(db2,0.555,RE)]:
        strip=_form_strip(d["matches"],8)
        sw=0.036; gap=0.004
        for i,(res,c,_) in enumerate(strip):
            x=xbase+i*(sw+gap)
            fig.add_artist(plt.Rectangle((x,0.30),sw,0.048,
                                          transform=fig.transFigure,
                                          facecolor=c,edgecolor="none",alpha=0.85))
            fig.text(x+sw/2,0.323+0.001,{"G":"W","B":"D","M":"L"}.get(res,res),ha="center",va="center",
                     fontsize=9,fontweight="bold",color=BG,transform=fig.transFigure)

    fig.text(0.085,0.36,"Last 8 Matches",fontsize=8,color=MU,transform=fig.transFigure)
    fig.text(0.555,0.36,"Last 8 Matches",fontsize=8,color=MU,transform=fig.transFigure)

    fig.text(0.5,0.06,"Footscrapping Analytics · match data",ha="center",
             fontsize=8,color=MU,transform=fig.transFigure)
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_metrics(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,f"Metric Comparison: {ta}  vs  {tb}",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(1,2,figure=fig,wspace=0.04,top=0.91,bottom=0.07,left=0.03,right=0.97)

    metrics=[
        ("xG / Match",       da["TXG"]/max(len(da["matches"]),1), db2["TXG"]/max(len(db2["matches"]),1), True),
        ("Shots on Target/Match", _ts_avg(da,"shots_on_target"),  _ts_avg(db2,"shots_on_target"),       True),
        ("Possession %",     _ts_avg(da,"ball_possession"),        _ts_avg(db2,"ball_possession"),       True),
        ("Pass Accuracy %",  _ts_avg(da,"accurate_passes_pct"),    _ts_avg(db2,"accurate_passes_pct"),   True),
        ("Key Passes/Match", _ts_avg(da,"key_passes"),            _ts_avg(db2,"key_passes"),            True),
        ("Corners/Match",    _ts_avg(da,"corner_kicks"),          _ts_avg(db2,"corner_kicks"),          True),
        ("Tackles/Match",    _ts_avg(da,"total_tackles"),         _ts_avg(db2,"total_tackles"),         True),
        ("Interceptions/Match", _ts_avg(da,"interceptions"),      _ts_avg(db2,"interceptions"),         True),
        ("xGA / Match",
         float(da["shots_ag"]["xg"].sum())/max(len(da["matches"]),1) if not da["shots_ag"].empty else 0,
         float(db2["shots_ag"]["xg"].sum())/max(len(db2["matches"]),1) if not db2["shots_ag"].empty else 0,
         False),  # lower = better
        ("Goals Conceded/Match", da["GA"]/max(len(da["matches"]),1), db2["GA"]/max(len(db2["matches"]),1), False),
        ("Fouls/Match",      _ts_avg(da,"fouls"),                 _ts_avg(db2,"fouls"),                 False),
    ]

    ax=fig.add_subplot(gs[0,:]); ax.set_facecolor(S1); ax.axis("off")

    # Header row
    ax.text(0.02,0.97,ta,transform=ax.transAxes,fontsize=10,
            fontweight="bold",color=BL,ha="left",va="top")
    ax.text(0.5,0.97,"Metric",transform=ax.transAxes,fontsize=9,
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

        # A bar (left → centre)
        bar_w=0.34
        ax.barh(y,frac_a*bar_w,left=0.5-frac_a*bar_w,height=row_h*0.55,
                color=BL if (higher_better and va2>=vb2) or (not higher_better and va2<=vb2) else "#2a3a4a",
                alpha=0.85,align="center",transform=ax.transAxes)
        # B bar (centre → right)
        ax.barh(y,frac_b*bar_w,left=0.5,height=row_h*0.55,
                color=RE if (higher_better and vb2>va2) or (not higher_better and vb2<va2) else "#4a2a2a",
                alpha=0.85,align="center",transform=ax.transAxes)

        # Values
        fmt=lambda v: f"{v:.2f}" if v<10 else f"{v:.1f}"
        ax.text(0.48,y,fmt(va2),transform=ax.transAxes,fontsize=8.5,
                color=TX,ha="right",va="center",fontweight="bold")
        ax.text(0.52,y,fmt(vb2),transform=ax.transAxes,fontsize=8.5,
                color=TX,ha="left",va="center",fontweight="bold")
        ax.text(0.5,y-row_h*0.28,lbl,transform=ax.transAxes,fontsize=7,
                color=MU,ha="center",va="center")

        # Divider
        ax.plot([0.01,0.99],[y-row_h*0.47,y-row_h*0.47],color=BR,lw=0.4,
                alpha=0.5,transform=ax.transAxes)

    _sp(ax); _border(ax)
    _cap(ax,"Per-match averages of both teams on 11 core metrics. The coloured bar marks the stronger side; values are written opposite each other in the middle of each row.")
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_shots(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    # vertical figsize (11×20) → pitch ratio (58:106≈0.547) fits the cells exactly
    fig=plt.figure(figsize=(11,20)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.985,"Shot Analysis — Attack & Defence",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(2,2,figure=fig,hspace=0.10,wspace=0.10,
                top=0.975,bottom=0.02,left=0.03,right=0.97)

    ax1=fig.add_subplot(gs[0,0])
    _shotmap(ax1,da["shots"],f"⚽ {ta} — Shots For"); _border(ax1)
    _cap(ax1,"Location of every shot. Size is proportional to xG; ★ goal, circle = saved/missed shot.")
    ax2=fig.add_subplot(gs[0,1])
    _heatmap(ax2,da["shots"],f"🔥 {ta} — Attacking Density",[OR,"#ff4500","#ff0000"]); _border(ax2)
    _cap(ax2,"KDE density map of shots for. % values are the share of shots taken from each zone.")
    ax3=fig.add_subplot(gs[1,0])
    _shotmap(ax3,db2["shots"],f"⚽ {tb} — Shots For"); _border(ax3)
    _cap(ax3,"Location of every shot. Size is proportional to xG; ★ goal, circle = saved/missed shot.")
    ax4=fig.add_subplot(gs[1,1])
    _heatmap(ax4,db2["shots"],f"🔥 {tb} — Attacking Density",[RE,OR,YE]); _border(ax4)
    _cap(ax4,"KDE density map of shots for. % values are the share of shots taken from each zone.")

    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


def _p_h2h_strategy(pdf, da:dict, db2:dict):
    ta,tb=da["team_name"],db2["team_name"]
    tips=_winning_tips(da,db2)
    ins_a=_insights(da); ins_b=_insights(db2)

    fig=plt.figure(figsize=(16,11)); fig.patch.set_facecolor(BG)
    fig.text(0.5,0.975,f"Tactical Analysis & Game Plan",ha="center",
             fontsize=12,fontweight="bold",color=TX)
    gs=GridSpec(2,3,figure=fig,hspace=0.55,wspace=0.35,
                top=0.92,bottom=0.07,left=0.04,right=0.97)

    # Radar comparison
    ax_r=fig.add_subplot(gs[0,:2],polar=True); ax_r.set_facecolor(S2)
    labels=["xG/Match","On Target","Possession%","Pass%","Corners","Tackles"]
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
    ax_r.set_title("Radar Comparison",fontsize=9,color=TX,pad=15)
    # (fig.text instead of _cap for the radar — polar axes do not support transAxes)
    fig.text(0.285,0.505,"Normalised values of both teams on 6 core metrics. Outer ring = best of the two, inner ring = weaker.",
             ha="center",fontsize=5.8,color=MU,style="italic")

    # Kazanma stratejisi metni
    ax_w=fig.add_subplot(gs[0,2]); ax_w.set_facecolor(S1); ax_w.axis("off")
    ax_w.set_title(f"Game Plan for {ta}",fontsize=9,color=GR,loc="left",pad=4)
    _sp(ax_w); _border(ax_w)
    y=0.92
    for tip in tips:
        for line in tip.split("\n"):
            ax_w.text(0.03,y,line,transform=ax_w.transAxes,fontsize=7.5,
                      color=TX,va="top"); y-=0.092
        y-=0.04
    _cap(ax_w,"Automatic tactical suggestions based on statistical advantages and opponent weaknesses. Inputs: xG, possession, set pieces and away record.")

    # Strengths (A)
    ax_ga=fig.add_subplot(gs[1,0]); ax_ga.set_facecolor(S1); ax_ga.axis("off")
    ax_ga.set_title(f"💪 {ta} Strengths",fontsize=9,color=BL,loc="left",pad=4)
    _sp(ax_ga); _border(ax_ga)
    y=0.88
    for ico,title,desc in [i for i in ins_a if i[0] in("✅","⭐","📈","⚽")][:4]:
        ax_ga.text(0.03,y,f"{ico} {title}",transform=ax_ga.transAxes,
                   fontsize=8,color=GR,fontweight="bold"); y-=0.12
        ax_ga.text(0.05,y,desc[:60],transform=ax_ga.transAxes,
                   fontsize=7,color=TX); y-=0.12
    _cap(ax_ga,"Statistical strengths where the first team performs above the league average and gains an edge on the pitch.")

    # Weaknesses (B)
    ax_wb=fig.add_subplot(gs[1,1]); ax_wb.set_facecolor(S1); ax_wb.axis("off")
    ax_wb.set_title(f"⚠️ {tb} Weaknesses",fontsize=9,color=RE,loc="left",pad=4)
    _sp(ax_wb); _border(ax_wb)
    y=0.88
    for ico,title,desc in [i for i in ins_b if i[0] in("⚠️","📉")][:4]:
        ax_wb.text(0.03,y,f"{ico} {title}",transform=ax_wb.transAxes,
                   fontsize=8,color=OR,fontweight="bold"); y-=0.12
        ax_wb.text(0.05,y,desc[:60],transform=ax_wb.transAxes,
                   fontsize=7,color=TX); y-=0.12
    _cap(ax_wb,"Statistical weak points of the opponent that need work and can be exploited.")

    # xG race
    ax_xg=fig.add_subplot(gs[1,2]); ax_xg.set_facecolor(S1)
    ax_xg.set_title("Cumulative xG Comparison",fontsize=9,color=TX,loc="left",pad=4)
    for d,color,label in [(da,BL,ta),(db2,RE,tb)]:
        if not d["xgd"].empty:
            cumxg=d["xgd"]["xg"].cumsum()
            ax_xg.plot(range(1,len(cumxg)+1),cumxg.values,
                       color=color,lw=2,label=label)
    ax_xg.legend(facecolor=S2,edgecolor=BR,fontsize=7)
    ax_xg.set_xlabel("Match",color=MU,fontsize=8)
    ax_xg.set_ylabel("Cumulative xG",color=MU,fontsize=8)
    ax_xg.grid(alpha=0.25); _sp(ax_xg); _border(ax_xg)
    _cap(ax_xg,"Total expected goals (xG) accumulated over the season. The higher team created more quality chances; the slope is the creation rate per match.")

    info=pdf.infodict()
    info["Title"]=f"{ta} vs {tb} Comparison"; info["Author"]="Footscrapping"
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

# ── "Why Successful?" page ─────────────────────────────────────────────────
def _p_why_successful(pdf, d: dict, team_stats_df):
    """Style fingerprint radar + strengths/weaknesses table page."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(0.5, 0.96, f"Why Successful? — {d['team_name']}", ha="center",
             fontsize=15, color=BL, fontweight="bold")

    if team_stats_df.empty:
        fig.text(0.5, 0.5, "No team statistics.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    try:
        fp = _sfp(team_stats_df, d["team_name"])
    except Exception:
        fig.text(0.5, 0.5, "Style fingerprint could not be computed.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    if fp.empty:
        fig.text(0.5, 0.5, "Team not found.", ha="center", color=MU, fontsize=11)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    sw = team_strengths_weaknesses(fp)

    # Left: radar
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
        ax_rad.set_title("League Percentile (%)", size=8, color=MU, pad=10)

    # Right: strengths/weaknesses table
    ax_tbl = fig.add_axes([0.50, 0.08, 0.48, 0.82])
    ax_tbl.set_facecolor(S1); ax_tbl.axis("off")
    _border(ax_tbl)

    ax_tbl.text(0.5, 0.97, "Strengths", ha="center", va="top",
                fontsize=10, color=GR, fontweight="bold", transform=ax_tbl.transAxes)
    strengths = sw.get("strengths", [])
    for i, s in enumerate(strengths[:6]):
        y = 0.90 - i * 0.07
        ax_tbl.text(0.05, y, f"✅ {s['label']} — {s['value']:.0f}. percentile",
                    va="top", fontsize=8.5, color=GR, transform=ax_tbl.transAxes)

    ax_tbl.text(0.5, 0.90 - len(strengths[:6]) * 0.07 - 0.04,
                "Weaknesses", ha="center", va="top",
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


# ── Head-to-head tactical suggestions page ────────────────────────────────
def _p_h2h_tactical(pdf, d_ours: dict, d_theirs: dict, team_stats_df):
    """Style overlay radar + automatic tactical suggestion texts for two teams."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.96, f"H2H — {d_ours['team_name']} vs {d_theirs['team_name']}",
             ha="center", fontsize=14, color=BL, fontweight="bold")

    if team_stats_df.empty:
        fig.text(0.5, 0.5, "No team statistics.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    try:
        fp_a = _sfp(team_stats_df, d_ours["team_name"])
        fp_b = _sfp(team_stats_df, d_theirs["team_name"])
    except Exception:
        fp_a, fp_b = None, None

    # Left: overlay radar
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

    # Right: tactical suggestion texts
    ax_txt = fig.add_axes([0.50, 0.08, 0.48, 0.82])
    ax_txt.set_facecolor(S1); ax_txt.axis("off"); _border(ax_txt)

    if fp_a is not None and not fp_a.empty and fp_b is not None and not fp_b.empty:
        reco = h2h_tactical_recommendations(fp_a, fp_b,
                                            our_name=d_ours["team_name"],
                                            their_name=d_theirs["team_name"])
        exploit = [r for r in reco if r.get("type") == "exploit"]
        beware  = [r for r in reco if r.get("type") == "beware"]

        ax_txt.text(0.5, 0.97, "How to Win?", ha="center", va="top",
                    fontsize=10, color=GR, fontweight="bold", transform=ax_txt.transAxes)
        for i, r in enumerate(exploit[:5]):
            y = 0.90 - i * 0.08
            ax_txt.text(0.04, y, f"✅ {r.get('text', '')}", va="top", fontsize=8,
                        color=GR, transform=ax_txt.transAxes, wrap=True)

        base = 0.90 - len(exploit[:5]) * 0.08 - 0.05
        ax_txt.text(0.5, base, "Watch Out For", ha="center", va="top",
                    fontsize=10, color=OR, fontweight="bold", transform=ax_txt.transAxes)
        for i, r in enumerate(beware[:5]):
            y = base - 0.06 - i * 0.08
            ax_txt.text(0.04, y, f"⚠️ {r.get('text', '')}", va="top", fontsize=8,
                        color=OR, transform=ax_txt.transAxes, wrap=True)
    else:
        ax_txt.text(0.5, 0.5, "Style data could not be computed.", ha="center",
                    color=MU, transform=ax_txt.transAxes)

    pdf.savefig(fig, dpi=120); plt.close(fig)


# ── Player detail page ─────────────────────────────────────────────────────
def _p_player_detail(pdf, player_row: dict, all_players_df):
    """Radar + metric table + similar players for a single player."""
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
    fig.text(0.5, 0.925, f"{pos} | {mac} matches | {mins} min | Avg. {rat:.2f}",
             ha="center", fontsize=9, color=MU)

    # Radar: p90 metrics vs league average
    metrics_raw = {
        "Goals":   float(player_row.get("gol") or 0),
        "Assists": float(player_row.get("ast") or 0),
        "xG":      float(player_row.get("xg") or 0),
        "xA":      float(player_row.get("xa") or 0),
        "Shots":   float(player_row.get("sut") or 0),
        "Passes":  float(player_row.get("pas") or 0),
        "Pressing":float(player_row.get("intp") or 0),
        "Duels":   float(player_row.get("dw") or 0),
        "Progression":float(player_row.get("prun") or 0),
    }
    # Per-90
    p90 = {k: v / mins * 90 if mins > 0 else 0 for k, v in metrics_raw.items()}

    if not all_players_df.empty:
        lig_avgs = {}
        col_map = {"Goals":"gol","Assists":"ast","xG":"xg","xA":"xa","Shots":"sut",
                   "Passes":"pas","Pressing":"intp","Duels":"dw","Progression":"prun"}
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
    ax_rad.set_title("League Percentile", size=8, color=MU, pad=8)

    # Metrik tablo
    ax_tbl = fig.add_axes([0.46, 0.40, 0.25, 0.50])
    ax_tbl.set_facecolor(S1); ax_tbl.axis("off"); _border(ax_tbl)
    ax_tbl.text(0.5, 0.97, "p90 Metrics", ha="center", va="top",
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
    ax_sim.text(0.5, 0.97, "Similar Players", ha="center", va="top",
                fontsize=9, color=TX, fontweight="bold", transform=ax_sim.transAxes)

    if not all_players_df.empty and "player_name" in all_players_df.columns:
        try:
            # Temporary DataFrame for a simple cosine similarity
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

    # Summary text
    ax_sum = fig.add_axes([0.46, 0.08, 0.52, 0.28])
    ax_sum.set_facecolor(S1); ax_sum.axis("off"); _border(ax_sum)
    top2_strong = sorted(pct.items(), key=lambda x: -x[1])[:2]
    top2_weak   = sorted(pct.items(), key=lambda x: x[1])[:2]
    lines = []
    for lbl, v in top2_strong:
        lines.append(f"✅ {lbl} — {v:.0f}th percentile in the league (strength)")
    for lbl, v in top2_weak:
        lines.append(f"⚠️ {lbl} — {v:.0f}th percentile in the league (to improve)")
    for i, line in enumerate(lines):
        y = 0.85 - i * 0.20
        color = GR if line.startswith("✅") else OR
        ax_sum.text(0.04, y, line, va="top", fontsize=8, color=color,
                    transform=ax_sum.transAxes)

    pdf.savefig(fig, dpi=120); plt.close(fig)


# ── Transfer target list page ─────────────────────────────────────────────
def _p_transfer_targets(pdf, team_name: str, player_pool, style_target: Optional[str] = None):
    """Missing positions in the squad + suggested transfer targets."""
    _rc()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.96, f"Transfer Targets — {team_name}", ha="center",
             fontsize=14, color=BL, fontweight="bold")
    if style_target:
        fig.text(0.5, 0.935, f"Style Target: {style_target}", ha="center",
                 fontsize=8.5, color=MU)

    if player_pool.empty:
        fig.text(0.5, 0.5, "No player pool.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    # Current squad
    team_df = player_pool[player_pool["team_name"].str.lower() == team_name.lower()]
    if team_df.empty:
        fig.text(0.5, 0.5, "No players found for this team.", ha="center", color=MU)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    # Simple squad-gap detection
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
        ax.text(0.5, 0.5, "The squad looks complete — no missing positions.",
                ha="center", va="center", fontsize=11, color=GR, transform=ax.transAxes)
        pdf.savefig(fig, dpi=120); plt.close(fig); return

    pos_labels = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}
    y = 0.93
    for pos_code, gap in targets_text:
        ax.text(0.03, y, f"📌 {pos_labels.get(pos_code, pos_code)} — {gap} needed",
                va="top", fontsize=10, color=OR, fontweight="bold", transform=ax.transAxes)
        y -= 0.05

        # Suggestions from players outside the team
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
        raise ValueError(f"No matches found in the database for '{team_name}'.")
    if output_path is None:
        safe = team_name.lower().replace(" ", "_")
        output_path = str(Path(tempfile.gettempdir()) / f"{safe}_rapor.pdf")
    ins = _insights(d)

    # Team statistics (for the style analysis)
    try:
        import pandas as pd
        team_stats_df = db.get_team_stats(None, season)
    except Exception:
        import pandas as pd
        team_stats_df = pd.DataFrame()

    # Opponent data (for H2H)
    d_opp = None
    if opponent:
        d_opp = _load(opponent, season)

    # Player pool (include_players + transfer targets)
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

        # "Why Successful?" page
        _p_why_successful(pdf, d, team_stats_df)

        # H2H page (if an opponent is given)
        if d_opp:
            _p_h2h_tactical(pdf, d, d_opp, team_stats_df)

        # Player detail pages
        if include_players and not d["players"].empty:
            for _, p_row in d["players"].iterrows():
                _p_player_detail(pdf, p_row.to_dict(), d["players"])

        # Transfer targets page
        _p_transfer_targets(pdf, team_name, player_pool, style_target=style_target)

    return output_path


def generate_comparison_report(team_a:str, team_b:str, season:Optional[str]=None,
                                output_path:Optional[str]=None)->str:
    _rc()
    da=_load(team_a,season); db2=_load(team_b,season)
    if not da: raise ValueError(f"No matches found for '{team_a}'.")
    if not db2: raise ValueError(f"No matches found for '{team_b}'.")
    if output_path is None:
        sa=team_a.lower().replace(" ","_"); sb=team_b.lower().replace(" ","_")
        output_path=str(Path(tempfile.gettempdir())/f"{sa}_vs_{sb}_rapor.pdf")
    with PdfPages(output_path) as pdf:
        _p_h2h_cover(pdf,da,db2)
        _p_h2h_metrics(pdf,da,db2)
        _p_h2h_shots(pdf,da,db2)
        _p_h2h_strategy(pdf,da,db2)
    return output_path
