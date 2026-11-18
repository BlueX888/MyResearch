/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

/**
 * 全局行业状态管理
 */
import { proxy, subscribe } from 'valtio'

// 行业配置类型
export interface IndustryConfig {
  id: string
  name: string
  description: string
  // 资讯搜索关键词
  newsKeywords: string[]
  // 招投标搜索关键词
  biddingKeywords: string[]
  // 研究相关关键词
  researchKeywords: string[]
}

// 预定义的行业配置
export const INDUSTRY_CONFIGS: IndustryConfig[] = [
  {
    id: 'aviation',
    name: '航空工业',
    description: '航空装备、主机厂（成飞/沈飞）、航发与产业链',
    newsKeywords: [
      '航空工业 十五五',
      '成飞 沈飞',
      '六代机',
      '舰载机',
      '航空发动机 国产化',
      '军贸 出口',
      '航空复合材料',
      '低空经济',
    ],
    biddingKeywords: [
      '航空装备',
      '飞机制造',
      '航空发动机',
      '机载系统',
      '航空复合材料',
      '航空维修',
    ],
    researchKeywords: ['成飞', '沈飞', '六代机', '航空发动机', '舰载机', '军贸出口'],
  },
]

// 行业状态
export interface IndustryState {
  currentIndustryId: string
  industries: IndustryConfig[]
}

// 从 localStorage 读取
const getStoredIndustryId = (): string => {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('selected_industry_id')
    console.log('[industry store] 从 localStorage 读取行业:', stored)
    // 行业被替换后，旧 id 回落到第一个
    if (stored && INDUSTRY_CONFIGS.some((i) => i.id === stored)) return stored
  }
  return INDUSTRY_CONFIGS[0].id
}

// 创建状态
export const industryState = proxy<IndustryState>({
  currentIndustryId: getStoredIndustryId(),
  industries: INDUSTRY_CONFIGS,
})

// 订阅变化，保存到 localStorage
subscribe(industryState, () => {
  if (typeof window !== 'undefined') {
    console.log('[industry store] 保存行业到 localStorage:', industryState.currentIndustryId)
    localStorage.setItem('selected_industry_id', industryState.currentIndustryId)
  }
})

// 获取当前行业配置
export const getCurrentIndustry = (): IndustryConfig => {
  const industry = industryState.industries.find(
    (i) => i.id === industryState.currentIndustryId
  )
  console.log('[industry store] 获取当前行业:', industry?.name)
  return industry || INDUSTRY_CONFIGS[0]
}

// 切换行业
export const setCurrentIndustry = (industryId: string) => {
  console.log('[industry store] 切换行业:', industryId)
  industryState.currentIndustryId = industryId
}

// 获取行业列表（用于选择器）
export const getIndustryOptions = () => {
  return industryState.industries.map((i) => ({
    value: i.id,
    label: i.name,
    description: i.description,
  }))
}
