/**
 * News Dashboard Module
 * Handles the real-time news feed panel with category filtering,
 * WebSocket updates, and Polymarket probability display.
 */

// News state
let newsItems = [];
let newsCurrentCategory = 'all';
let newsPanelActive = false;
let newsUnreadCount = 0;

// Source icons
const SOURCE_ICONS = {
    'Alpaca': 'A',
    'Benzinga': 'B',
    'X.com': 'X',
    'Polymarket': 'P',
};

const CATEGORY_COLORS = {
    'tech': '#2962FF',
    'financial': '#FF9800',
    'geopolitical': '#ef5350',
    'prediction_market': '#ab47bc',
};

const SENTIMENT_SYMBOLS = {
    'bullish': '+',
    'bearish': '-',
    'neutral': '',
};

/**
 * Initialize the news dashboard — toggle buttons, category tabs, initial REST load.
 */
function initNewsDashboard() {
    // Panel toggle buttons
    document.querySelectorAll('.panel-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.dataset.panel;
            switchPanelMode(panel);
        });
    });

    // Category filter tabs
    document.querySelectorAll('.news-cat-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cat = btn.dataset.category;
            filterNewsByCategory(cat);
        });
    });

    // Load initial news from REST endpoint
    loadInitialNews();
}

/**
 * Switch between Stats and News views.
 */
function switchPanelMode(mode) {
    // Update toggle buttons
    document.querySelectorAll('.panel-toggle-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.panel === mode);
    });

    // Update panel views
    document.querySelectorAll('.panel-view').forEach(view => {
        view.classList.remove('active');
    });

    const target = document.getElementById(`view-${mode}`);
    if (target) target.classList.add('active');

    newsPanelActive = (mode === 'news');

    // Reset unread count when switching to news
    if (newsPanelActive && newsUnreadCount > 0) {
        newsUnreadCount = 0;
        updateNewsBadge();
        // Tell server we've seen the items
        if (typeof socket !== 'undefined' && socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'news_ack' }));
        }
    }
}

/**
 * Load initial news items via REST API.
 */
async function loadInitialNews() {
    try {
        const resp = await fetch('/api/news/feed?limit=50');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.items && data.items.length > 0) {
            newsItems = data.items;
            renderNewsFeed();
            updateNewsStatus(data.items.length);
        }
    } catch (e) {
        console.warn('Failed to load initial news:', e);
    }
}

/**
 * Handle incoming news_update WebSocket message.
 */
function handleNewsUpdate(data) {
    if (!data.items || data.items.length === 0) return;

    // Prepend new items (they come newest first)
    const existingIds = new Set(newsItems.map(i => i.id));
    const newItems = data.items.filter(i => !existingIds.has(i.id));

    if (newItems.length === 0) return;

    newsItems = [...newItems, ...newsItems].slice(0, 200);

    // Update unread badge (only if not actively viewing news)
    if (!newsPanelActive) {
        newsUnreadCount += newItems.length;
        updateNewsBadge();
    }

    renderNewsFeed();
    updateNewsStatus(data.total_count || newsItems.length, data.timestamp);
}

/**
 * Filter displayed news by category.
 */
function filterNewsByCategory(category) {
    newsCurrentCategory = category;

    // Update active tab
    document.querySelectorAll('.news-cat-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.category === category);
    });

    renderNewsFeed();
}

/**
 * Render the news feed based on current filter.
 */
function renderNewsFeed() {
    const feed = document.getElementById('news-feed');
    if (!feed) return;

    let items = newsItems;
    if (newsCurrentCategory !== 'all') {
        items = items.filter(i => i.category === newsCurrentCategory);
    }

    if (items.length === 0) {
        feed.innerHTML = `
            <div class="news-empty-state">
                <p>${newsCurrentCategory !== 'all' ? 'No items in this category yet.' : 'Monitoring news sources...'}</p>
                <p class="news-empty-sub">Items will appear as they arrive from Alpaca, X.com, and Polymarket.</p>
            </div>`;
        return;
    }

    feed.innerHTML = items.map(renderNewsItem).join('');
}

/**
 * Render a single news item.
 */
function renderNewsItem(item) {
    const sourceIcon = SOURCE_ICONS[item.source] || item.source?.[0] || '?';
    const categoryColor = CATEGORY_COLORS[item.category] || '#9598a1';
    const categoryLabel = (item.category || 'tech').toUpperCase().replace('_', ' ').slice(0, 3);
    const sentiment = SENTIMENT_SYMBOLS[item.sentiment] || '';
    const timeAgo = formatTimeAgo(item.created_at);
    const headline = escapeHtml(item.headline || '');
    const summary = escapeHtml(item.summary || '');
    const symbols = (item.symbols || []).join(', ');

    // Polymarket-specific rendering
    if (item.source === 'Polymarket') {
        return renderPolymarketItem(item, sourceIcon, timeAgo);
    }

    // News / Twitter item
    const sentimentClass = item.sentiment === 'bullish' ? 'news-sentiment-bull' :
                           item.sentiment === 'bearish' ? 'news-sentiment-bear' : '';

    return `
        <div class="news-item" data-id="${item.id}">
            <div class="news-item-header">
                <span class="news-source-icon" style="background:${categoryColor}">${sourceIcon}</span>
                <span class="news-time">${timeAgo}</span>
                <span class="news-category-tag" style="color:${categoryColor}">${categoryLabel}</span>
                ${sentiment ? `<span class="news-sentiment ${sentimentClass}">${sentiment}</span>` : ''}
            </div>
            <div class="news-item-headline">${headline}</div>
            ${symbols ? `<div class="news-item-symbols">${symbols}</div>` : ''}
            ${item.url ? `<div class="news-item-meta"><a href="${item.url}" target="_blank" rel="noopener">Open</a>${item.author ? ` &middot; ${escapeHtml(item.author)}` : ''}</div>` : ''}
        </div>`;
}

/**
 * Render a Polymarket prediction market item.
 */
function renderPolymarketItem(item, sourceIcon, timeAgo) {
    const headline = escapeHtml(item.headline || '');
    const prob = item.probability;
    const probPct = prob != null ? Math.round(prob * 100) : null;
    const volume = item.volume ? formatVolume(item.volume) : null;

    return `
        <div class="news-item news-item-polymarket" data-id="${item.id}">
            <div class="news-item-header">
                <span class="news-source-icon" style="background:#ab47bc">${sourceIcon}</span>
                <span class="news-time">${timeAgo}</span>
                <span class="news-category-tag" style="color:#ab47bc">MKT</span>
            </div>
            <div class="news-item-headline">${headline}</div>
            ${probPct != null ? `
            <div class="news-poly-prob">
                <div class="news-poly-bar-bg">
                    <div class="news-poly-bar-fill" style="width:${probPct}%"></div>
                </div>
                <span class="news-poly-pct">${probPct}% Yes</span>
                ${volume ? `<span class="news-poly-vol">$${volume}</span>` : ''}
            </div>` : ''}
            ${item.url ? `<div class="news-item-meta"><a href="${item.url}" target="_blank" rel="noopener">View on Polymarket</a></div>` : ''}
        </div>`;
}

/**
 * Update the unread badge on the News toggle button.
 */
function updateNewsBadge() {
    const badge = document.getElementById('news-badge');
    if (!badge) return;

    if (newsUnreadCount > 0) {
        badge.textContent = newsUnreadCount > 99 ? '99+' : newsUnreadCount;
        badge.style.display = 'inline-flex';
    } else {
        badge.style.display = 'none';
    }
}

/**
 * Update the status bar in the news header.
 */
function updateNewsStatus(count, timestamp) {
    const countEl = document.getElementById('news-item-count');
    const pollEl = document.getElementById('news-last-poll');
    const dotEl = document.getElementById('news-live-dot');

    if (countEl) countEl.textContent = `${count} item${count !== 1 ? 's' : ''}`;
    if (pollEl && timestamp) pollEl.textContent = formatTimeAgo(timestamp);
    if (dotEl) dotEl.classList.add('active');
}

// ---- Utility Functions ----

function formatTimeAgo(isoString) {
    if (!isoString) return '';
    const then = new Date(isoString);
    const now = new Date();
    const diffMs = now - then;

    if (diffMs < 0) return 'just now';

    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;

    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;

    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

function formatVolume(vol) {
    if (vol >= 1e6) return (vol / 1e6).toFixed(1) + 'M';
    if (vol >= 1e3) return (vol / 1e3).toFixed(1) + 'K';
    return Math.round(vol).toString();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
