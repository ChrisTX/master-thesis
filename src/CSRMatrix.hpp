#ifndef GUARD_CSR_MATRIX_HPP
#define GUARD_CSR_MATRIX_HPP

#include <iosfwd>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <mkl.h>

#include "Utility.hpp"

namespace Utility {

	struct MatrixNaturalIndexing {};
	static const MatrixNaturalIndexing naturalindex{};

	template<typename T>
	class CSRMatrix {
	public:
		using value_type = T;

		// A std::valarray is indexed using a size_t type by definition (§iso.26.6.2.1)
		using size_type = MKL_INT;
		using vector_type = std::vector<value_type>;
		using index_vector_type = std::vector<size_type>;

		// Note that the iterator is ALWAYS constant!
		class CSRMatrixIter : public std::bidirectional_iterator_tag {
			const CSRMatrix* m_host;
			size_type m_pos;

		public:
			using difference_type = std::size_t;
			using value_type = T;
			using pointer = typename std::add_pointer_t<value_type>;
			using reference = typename std::add_lvalue_reference_t<value_type>;
			using iterator_category = std::bidirectional_iterator_tag;

			CSRMatrixIter(const CSRMatrix* host, size_type pos) : m_host{ host }, m_pos{ pos } { }

			inline value_type operator*() const
			{
				assert(m_pos < m_host->m_Entries.size());
				return m_host->m_Entries[m_pos];
			}

			CSRMatrixIter& operator++()
			{
				m_pos++;
				return *this;
			}

			inline CSRMatrixIter operator++(int)
			{
				CSRMatrixIter clone(*this);
				this->operator++();
				return clone;
			}

			CSRMatrixIter& operator--()
			{
				m_pos--;
				return *this;
			}

			inline CSRMatrixIter operator--(int)
			{
				CSRMatrixIter clone(*this);
				this->operator--();
				return clone;
			}

			inline bool operator==(const CSRMatrixIter& other) const
			{
				return (other.m_host == m_host && other.m_pos == m_pos);
			}

			inline bool operator!=(const CSRMatrixIter& other) const
			{
				return !(this->operator==(other));
			}

			inline std::enable_if<std::is_class<value_type>::value, pointer> operator->() const
			{
				assert(m_pos < m_host->m_Entries.size());
				return &m_host->m_Entries[m_pos];
			}

			inline size_type GetColumn() const
			{
				assert(m_pos < m_host->m_ColumnIndices.size());
				return m_host->m_ColumnIndices[m_pos] - 1;
			}
		};

		using const_iterator = CSRMatrixIter;

		// Standard constructors to ensure default behaviors
		CSRMatrix(size_type Rows, size_type Columns) : m_Rows{ Rows }, m_Columns{ Columns }, m_RowIndices{ Rows } { }

		CSRMatrix(size_type Rows, size_type Columns, vector_type Entries, index_vector_type ColumnIndices, index_vector_type RowIndices) :
			m_Rows{ Rows }, m_Columns{ Columns }, m_Entries( Entries ), m_ColumnIndices( ColumnIndices ), m_RowIndices( std::move(RowIndices) )
		{
			assert(m_ColumnIndices.size() == m_Entries.size() && (static_cast<std::size_t>(m_Rows) + 1) == m_RowIndices.size());
		}

		inline size_type GetNumberOfRows() const
		{
			return this->m_Rows;
		}

		inline size_type GetNumberOfColumns() const
		{
			return this->m_Columns;
		}

		inline const_iterator BeginRow(size_type Row) const
		{
			assert(Row < m_Rows);
			return const_iterator(this, m_RowIndices[Row] - 1);
		}

		inline const_iterator BeginRow(size_type Row, const MatrixNaturalIndexing) const
		{
			assert(Row);
			return BeginRow(Row - 1);
		}

		inline const_iterator EndRow(size_type Row) const
		{
			assert(Row < m_Rows);
			size_type rowend;

			if (Row + 1 < m_Rows)
				rowend = m_RowIndices[Row + 1] - 1;
			else
				rowend = GetNumberOfEntries();

			return const_iterator(this, rowend);
		}

		inline const_iterator EndRow(size_type Row, const MatrixNaturalIndexing) const
		{
			assert(Row);
			return EndRow(Row - 1);
		}

		inline T& FindEntry(size_type Row, size_type Column)
		{
			return m_Entries[FindEntryIndex(Row, Column)];
		}

		inline const T& FindEntry(size_type Row, size_type Column) const
		{
			return m_Entries[FindEntryIndex(Row, Column)];
		}

		inline T operator()(size_type Row, size_type Column) const
		{
			try {
				return FindEntry(Row, Column);
			}
			catch(std::out_of_range) {
				return T{};
			}
		}

		inline T operator()(size_type Row, size_type Column, const MatrixNaturalIndexing) const
		{
			assert(Row && Column);
			return this->operator()(Row - 1, Column - 1);
		}

		inline const T* GetData() const
		{
			return m_Entries.data();
		}

		inline size_type GetNumberOfEntries() const
		{
			return m_Entries.size();
		}

		template<typename target_value>
		friend std::ostream& operator<< (std::ostream& os, const CSRMatrix<target_value>& csrmat);

		template<typename K>
		inline auto operator*(const K& vec) const
		{
			assert(m_Columns == vec.size());
			std::vector<value_type> resvec(m_Rows);

			char transa = 'N';
			Utility::MKL_csrgemv(&transa, const_cast<MKL_INT*>(&m_Rows), const_cast<value_type*>(m_Entries.data()), const_cast<MKL_INT*>(m_RowIndices.data()), const_cast<MKL_INT*>(m_ColumnIndices.data()), const_cast<value_type*>(vec.data()), resvec.data());

			return resvec;
		}

	protected:
		const size_type m_Rows;
		const size_type m_Columns;

	public:
		vector_type m_Entries;
		index_vector_type m_RowIndices;
		index_vector_type m_ColumnIndices;

	protected:
		std::size_t FindEntryIndex(size_type Row, size_type Column) const
		{
			assert(Row < m_Rows && Column < m_Columns);
			auto StartIterator = m_ColumnIndices.cbegin();
			std::advance(StartIterator, m_RowIndices[Row] - 1);
			
			auto EndIterator = m_ColumnIndices.cbegin();
			std::advance(EndIterator, m_RowIndices[Row + 1] - 1);

			const auto lowerFind = std::lower_bound(StartIterator, EndIterator, Column + 1);
			if (lowerFind != EndIterator && *lowerFind == Column + 1) {
				const auto diffAmount = lowerFind - m_ColumnIndices.cbegin();
				assert(diffAmount >= 0);
				return static_cast<std::size_t>(diffAmount);
			}

			throw std::out_of_range("Entry is zero.");
		}
	};

	template<typename T>
	std::ostream& operator<< (std::ostream& os, const CSRMatrix<T>& csrmat)
	{
		using size_type = typename CSRMatrix<T>::size_type;

		std::ostream::sentry s(os);
		if (!s)
			return os;

		for (size_type i = 0; i < csrmat.m_Rows; ++i) {
			for (size_type j = 0; j < csrmat.m_Columns; ++j)
				os << std::setw(16) << csrmat(i, j);

			os << '\n';
		}

		return os;
	}

	template<typename T>
	class CSRMatrixAssembler {
	public:
		using value_type = T;
		using size_type = MKL_INT;

		CSRMatrixAssembler(size_type Rows, size_type Columns) : m_Rows{ Rows }, m_Columns{ Columns }
		{ }

		void EraseRow(size_type Row)
		{
			assert(Row < m_Rows);
			m_EntriesBuildUpMap.erase(Row);
		}

		inline T& operator()(size_type Row, size_type Column)
		{
			assert(0 <= Row && 0 <= Column && Row < m_Rows && Column < m_Columns);
#ifdef SYMMETRIC_ASSEMBLY
			assert(Column >= Row);
#endif
			return m_EntriesBuildUpMap[Row][Column];
		}

		inline T operator()(size_type Row, size_type Column) const
		{
			assert(0 <= Row && 0 <= Column && Row < m_Rows && Column < m_Columns);
#ifdef SYMMETRIC_ASSEMBLY
			assert(Column >= Row);
#endif
			return m_EntriesBuildUpMap.at(Row).at(Column);
		}

		CSRMatrix<T> AssembleMatrix(const T epsilon_filter = T{ 0 }) const
		{
			// Now we have our map loaded with pairs i,j mapped to their values
			// Note that a std::pair is compared lexographically in its elements. This means that if we iterate through the map,
			// we will pass the first entry in ascending order and then the second entry.

			auto AmountOfNonZeroEntries = size_type{ 0 };
			for (auto it = m_EntriesBuildUpMap.cbegin(); it != m_EntriesBuildUpMap.cend(); ++it)
				AmountOfNonZeroEntries += static_cast<size_type>( it->second.size() );

			size_type RowCounter = 0;
			size_type EntryPosition = 0;

			using CSR_vector_type = typename CSRMatrix<T>::vector_type;
			using CSR_index_vector_type = typename CSRMatrix<T>::index_vector_type;

			CSR_index_vector_type RowIndices(m_Rows + 1);
			CSR_vector_type Entries(AmountOfNonZeroEntries);
			CSR_index_vector_type ColumnIndices(AmountOfNonZeroEntries);

			// Special case, pulled from the loop
			RowIndices[0] = 1;

			for (auto it = m_EntriesBuildUpMap.cbegin(); it != m_EntriesBuildUpMap.cend(); ++it) {
				const auto i = it->first;

				assert(!i || i == RowCounter + 1);
				RowCounter = i;
				
				for (auto colit = it->second.cbegin(); colit != it->second.cend(); ++colit) {
					if (colit->first != RowCounter && std::abs(colit->second) < epsilon_filter)
						continue;
					Entries[EntryPosition] = colit->second;
					assert(std::isfinite(colit->second));
					ColumnIndices[EntryPosition++] = colit->first + 1;
#ifndef NDEBUG
					if (EntryPosition > RowIndices[RowCounter])
						assert(ColumnIndices[EntryPosition - 1] > ColumnIndices[EntryPosition - 2]);
#endif
				}
				RowIndices[i + 1] = EntryPosition + 1;
			}
			assert(RowIndices[m_Rows] == EntryPosition + 1);

			return CSRMatrix<T>{ m_Rows, m_Columns, std::move(Entries), std::move(ColumnIndices), std::move(RowIndices) };
		}

		std::vector<T> AssembleDenseMatrix() const
		{
			std::vector<T> DenseMatrix(m_Rows * m_Columns, T{ 0 });

			for (auto it = m_EntriesBuildUpMap.cbegin(); it != m_EntriesBuildUpMap.cend(); ++it) {
				const auto i = it->first;

				for (auto colit = it->second.cbegin(); colit != it->second.cend(); ++colit)
					DenseMatrix[i * m_Columns + colit->first] = colit->second;
			}

			return DenseMatrix;
		}

		void ResetRow(const size_type row)
		{
			assert(0 <= row && row < m_Rows);
			m_EntriesBuildUpMap[row].clear();
		}

		void ResetColumn(const size_type col, const bool symmetric_matrix)
		{
			auto i_end = m_Columns;
			if (symmetric_matrix)
				i_end = col;
			for (auto i = size_type{ 0 }; i < i_end; ++i) {
				m_EntriesBuildUpMap[i].erase(col);
			}

		}

		auto GetNumberOfRows() const
		{
			return m_Rows;
		}

		auto GetNumberOfColumns() const
		{
			return m_Rows;
		}

	protected:
		std::map<size_type, std::map<size_type, T>> m_EntriesBuildUpMap;
		const size_type m_Rows;
		const size_type m_Columns;
	};
}

#endif
