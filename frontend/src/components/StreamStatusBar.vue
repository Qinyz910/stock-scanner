<template>
  <div class="stream-status">
    <div class="stream-status__header">
      <n-text size="small" class="stream-status__status">{{ statusMessage }}</n-text>
      <n-button
        v-if="canCancel"
        size="tiny"
        tertiary
        type="error"
        @click="emit('cancel')"
      >
        取消
      </n-button>
    </div>
    <n-progress
      type="line"
      :status="isStreaming ? 'info' : 'success'"
      :percentage="Math.min(percentage, 100)"
      :processing="isStreaming"
    />
    <n-space size="small" class="stream-status__meta">
      <span>进度: {{ processed }}/{{ totalLabel }}</span>
      <span v-if="chunkCount">数据块: {{ chunkCount }}</span>
      <span v-if="retries">重试: {{ retries }}</span>
    </n-space>
  </div>
</template>

<script setup lang="ts">
import { computed, toRefs } from 'vue';
import { NProgress, NButton, NSpace, NText } from 'naive-ui';

const props = defineProps<{
  isStreaming: boolean;
  processed: number;
  total: number;
  chunkCount: number;
  retries: number;
  statusText: string;
  canCancel: boolean;
}>();

const emit = defineEmits<{
  (e: 'cancel'): void;
}>();

const { isStreaming, processed, total, chunkCount, retries, statusText, canCancel } = toRefs(props);

const totalLabel = computed(() => (total.value > 0 ? total.value : '未知'));

const percentage = computed(() => {
  if (total.value > 0) {
    const ratio = processed.value / total.value;
    const base = Math.min(100, Math.round(ratio * 100));
    return isStreaming.value ? Math.min(base, 95) : base;
  }
  if (isStreaming.value) {
    return Math.min(95, 10 + chunkCount.value * 5);
  }
  return chunkCount.value > 0 ? 100 : 0;
});

const statusMessage = computed(() => statusText.value || (isStreaming.value ? '处理中...' : '准备就绪'));
</script>

<style scoped>
.stream-status {
  margin: 12px 0 16px;
  padding: 12px 16px;
  background-color: var(--n-color);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(31, 41, 55, 0.08);
}

.stream-status__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.stream-status__status {
  font-weight: 500;
}

.stream-status__meta {
  font-size: 12px;
  color: var(--n-text-color-3);
}
</style>
