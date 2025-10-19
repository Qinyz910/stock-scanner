<template>
  <n-drawer v-model:show="internalShow" :width="isMobile ? '100%' : 480" placement="right" :trap-focus="false">
    <n-drawer-content :title="drawerTitle">
      <div v-if="!code" class="empty-wrap">
        <n-empty description="未选择股票" size="large" />
      </div>
      <template v-else>
        <div v-if="loading" class="loading-wrap">
          <n-skeleton height="16px" :repeat="6" animated />
        </div>
        <div v-else-if="error">
          <n-empty :description="error" size="large" />
        </div>
        <div v-else-if="!scoreDetail">
          <n-empty description="暂无评分数据" size="large" />
        </div>
        <div v-else class="content-wrap">
          <div class="score-header">
            <div class="score-main">
              <div class="score-value">
                综合评分: <span class="number">{{ scoreDetail.score }}</span>
              </div>
              <n-tag v-if="scoreDetail.confidence !== undefined" type="success" size="small" round>
                置信度 {{ formatPercent(scoreDetail.confidence) }}
              </n-tag>
            </div>
            <div class="header-actions">
              <n-button tertiary size="small" @click="refresh" :loading="loading">刷新</n-button>
            </div>
          </div>
          <n-divider dashed>因子贡献</n-divider>
          <div v-if="normalizedContributions.length === 0">
            <n-empty description="暂无贡献明细" size="large" />
          </div>
          <div v-else class="contrib-list">
            <div
              v-for="item in normalizedContributions"
              :key="item.factor"
              class="contrib-item"
            >
              <div class="factor-label">{{ item.factor }}</div>
              <div class="bar-wrap">
                <div class="bar-bg">
                  <div
                    class="bar-fill"
                    :class="{ positive: item.value >= 0, negative: item.value < 0 }"
                    :style="{ width: computeWidth(item.value) }"
                  />
                </div>
              </div>
              <div class="value-label" :class="{ up: item.value > 0, down: item.value < 0 }">
                {{ formatNumber(item.value) }}
              </div>
            </div>
          </div>
        </div>
      </template>
    </n-drawer-content>
  </n-drawer>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { NDrawer, NDrawerContent, NEmpty, NSkeleton, NTag, NDivider, NButton } from 'naive-ui';
import { apiService } from '@/services/api';

interface ScoreContribution {
  factor: string;
  value: number;
}

interface ScoreItem {
  code: string;
  score: number;
  confidence?: number;
  contributions?: ScoreContribution[] | Record<string, number> | null;
}

interface ScoresResponse {
  items: ScoreItem[];
  total?: number;
  page?: number;
  page_size?: number;
}

const props = defineProps<{
  show: boolean;
  code: string | null;
  // 可注入，用于测试
  scoresLoader?: (codes: string[]) => Promise<ScoresResponse>;
}>();

const emit = defineEmits<{
  (e: 'update:show', value: boolean): void;
}>();

const isMobile = computed(() => window.innerWidth <= 768);
const internalShow = computed({
  get: () => props.show,
  set: (val: boolean) => emit('update:show', val)
});

const loading = ref(false);
const error = ref<string | null>(null);
const scoreDetail = ref<ScoreItem | null>(null);

const drawerTitle = computed(() => props.code ? `贡献解释 - ${props.code}` : '贡献解释');

function pickFirst(items: ScoreItem[]): ScoreItem | null {
  if (!Array.isArray(items) || items.length === 0) return null;
  return items.find(i => i.code === props.code) || items[0] || null;
}

async function load() {
  if (!props.code) {
    scoreDetail.value = null;
    return;
  }
  loading.value = true;
  error.value = null;
  try {
    const loader = props.scoresLoader || (async (codes: string[]) => apiService.getScores({ codes }));
    const res = await loader([props.code]);
    scoreDetail.value = pickFirst(res.items);
  } catch (e: any) {
    error.value = e?.message || '获取评分失败';
    scoreDetail.value = null;
  } finally {
    loading.value = false;
  }
}

function refresh() {
  load();
}

watch(() => props.show, (show) => {
  if (show) {
    load();
  }
});

watch(() => props.code, () => {
  if (props.show) {
    load();
  } else {
    scoreDetail.value = null;
    error.value = null;
  }
});

function normalizeContribs(contribs?: ScoreItem['contributions']): ScoreContribution[] {
  if (!contribs) return [];
  if (Array.isArray(contribs)) return contribs.filter(i => typeof i.value === 'number');
  return Object.entries(contribs)
    .map(([factor, value]) => ({ factor, value: Number(value as number) }))
    .filter(i => !Number.isNaN(i.value));
}

const normalizedContributions = computed(() => {
  return normalizeContribs(scoreDetail.value?.contributions).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
});

const maxAbs = computed(() => {
  const arr = normalizedContributions.value;
  return arr.length ? Math.max(...arr.map(i => Math.abs(i.value))) : 1;
});

function computeWidth(v: number): string {
  const percent = maxAbs.value ? Math.min(100, Math.round(Math.abs(v) / maxAbs.value * 100)) : 0;
  return percent + '%';
}

function formatNumber(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function formatPercent(v: number): string {
  if (v <= 1) return Math.round(v * 100) + '%';
  return Math.round(v) + '%';
}
</script>

<style scoped>
.empty-wrap, .loading-wrap {
  padding: 8px 4px;
}

.content-wrap {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.score-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.score-main {
  display: flex;
  align-items: center;
  gap: 8px;
}

.score-value .number {
  font-weight: 700;
}

.contrib-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.contrib-item {
  display: grid;
  grid-template-columns: 110px 1fr 60px;
  align-items: center;
  gap: 8px;
}

.factor-label {
  color: var(--n-text-color);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bar-wrap {
  width: 100%;
}

.bar-bg {
  width: 100%;
  height: 8px;
  background: var(--n-border-color);
  border-radius: 999px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  transition: width .2s ease;
}

.bar-fill.positive {
  background: rgba(24, 160, 88, .85);
}

.bar-fill.negative {
  background: rgba(208, 48, 80, .85);
}

.value-label {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.value-label.up {
  color: #18a058;
}

.value-label.down {
  color: #d03050;
}
</style>
